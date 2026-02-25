#include "cudaWarpCompressDouble.cuh"
#include "cudaTimer.cuh"
#include "utils.cuh"
#include "cudaUtils.cuh"
#include <chrono>
#ifdef GPZ_USE_NVCOMP
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#endif
#include "CpuTimer.hpp"
#include <nvtx3/nvToolsExt.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <utility>

#ifdef GPZ_USE_NVCOMP
using namespace nvcomp;
#endif

constexpr uint64_t META_DATA_BYTES = 64;

CudaWarpCompressDouble::CudaWarpCompressDouble(CudaConfig cudaConfig) : cuda_config(std::move(cudaConfig)) {
}

CudaWarpCompressDouble::CudaWarpCompressDouble() = default;

CudaWarpCompressDouble::~CudaWarpCompressDouble()
= default;

namespace WarpCompressorDouble {
    __global__ void
    runCompactKernel(const unsigned char *const __restrict__ comp_buff,
                     unsigned char *const __restrict__ comp_data,
                     const uint64_t *const __restrict__ comp_offset, const uint64_t *const __restrict__ d_loc_offset) {
        // fixed region size, elements * 8 bytes * 3 parts
        constexpr uint64_t BLOCK_BYTES = ELEMS_PER_BLOCK * sizeof(uint64_t) * 3;
        using word_t = uint32_t;

        // vector size: 4 bytes
        constexpr uint64_t WORD_SIZE = sizeof(word_t);
        const uint64_t block_id = blockIdx.x;

        const uint64_t valid_bytes = d_loc_offset[block_id];
        const uint64_t valid_words = (valid_bytes + WORD_SIZE - 1) / WORD_SIZE;

        const uint64_t src_offset = block_id * BLOCK_BYTES;
        const uint64_t dst_offset = comp_offset[block_id];

        const word_t *src = reinterpret_cast<const word_t *>(comp_buff + src_offset);
        word_t *dst = reinterpret_cast<word_t *>(comp_data + dst_offset);

        for (uint32_t i = threadIdx.x; i < valid_words; i += blockDim.x) {
            dst[i] = src[i];
        }
    }

    __global__ void
    runCompressKernel(const double *const __restrict__ d_x, const double *const __restrict__ d_y,
                      const double *const __restrict__ d_z,
                      unsigned char *const __restrict__ comp_data,
                      volatile uint64_t *const __restrict__ loc_offset,
                      uint32_t seg_size, uint64_t coordinate_num, double eb) {
        __shared__ double min_max[6];
        __shared__ uint64_t outlier;

        using BlockRadixSort = cub::BlockRadixSort<uint64_t, BLOCK_SIZE, CHUNK_SIZE, uint64_t>;
        union SharedMemory {
            // Temporary storage for sorting
            typename BlockRadixSort::TempStorage sort_temp_storage;
            uint64_t boundary_idx[ELEMS_PER_BLOCK];
            uint64_t b_delta_bids[ELEMS_PER_BLOCK];
            uint64_t b_rel_pos[BLOCK_SIZE][CHUNK_SIZE + 1];
        };
        __shared__ SharedMemory smem;

        __shared__ uint16_t max_bits_unique_num[4];
        __shared__ uint16_t b_counts[ELEMS_PER_BLOCK];
        __shared__ unsigned char sbuf[CHUNK_SIZE][BLOCK_SIZE + 1];

        const uint64_t meta_offset = gridDim.x * META_DATA_BYTES;

        uint32_t tid = threadIdx.x;
        uint32_t bid = blockIdx.x;
        uint32_t bid_size = blockDim.x;
        uint32_t warp_id = (bid * bid_size + tid) >> 5;

        int32_t local_block_max_block_id_bit = -INT_MAX;
        int32_t local_max_delta_block_id_bit = -INT_MAX;
        int32_t local_max_cnt_bit = -INT_MAX;
        int32_t local_max_rel_pos_bit = -INT_MAX;

        uint64_t local_block_ids[CHUNK_SIZE];
        uint64_t local_rel_pos[CHUNK_SIZE];
        uint32_t local_prefix_sum[CHUNK_SIZE];

        // find min max within warp
        double3 local_min = make_double3(DBL_MAX, DBL_MAX, DBL_MAX);
        double3 local_max = make_double3(-DBL_MAX, -DBL_MAX, -DBL_MAX);

        double x_coords[CHUNK_SIZE];
        double y_coords[CHUNK_SIZE];
        double z_coords[CHUNK_SIZE];
        uint64_t warp_base_start_idx = warp_id * ELEMS_PER_BLOCK;
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
            uint64_t index = warp_base_start_idx + i * BLOCK_SIZE + tid;

            bool valid = index < coordinate_num;

            double x = 0.0f, y = 0.0f, z = 0.0f;
            if (valid) {
                x = d_x[index];
                y = d_y[index];
                z = d_z[index];
            }
            x_coords[i] = x;
            y_coords[i] = y;
            z_coords[i] = z;

            // only valid points are used to calculate min/max
            if (valid) {
                local_min.x = fmin(local_min.x, x);
                local_min.y = fmin(local_min.y, y);
                local_min.z = fmin(local_min.z, z);

                local_max.x = fmax(local_max.x, x);
                local_max.y = fmax(local_max.y, y);
                local_max.z = fmax(local_max.z, z);
            }
        }

        // get min/max within warp and broadcast
        local_min.x = warpReduceMin(local_min.x);
        local_min.y = warpReduceMin(local_min.y);
        local_min.z = warpReduceMin(local_min.z);

        local_max.x = warpReduceMax(local_max.x);
        local_max.y = warpReduceMax(local_max.y);
        local_max.z = warpReduceMax(local_max.z);

        double inv_2eb = 1.0f / (2.0f * eb);
        uint64_t block_num_x = 0, block_num_y = 0;
        uint32_t seg_shift = 0;
        if (tid == 0) {
            min_max[0] = local_min.x;
            min_max[1] = local_min.y;
            min_max[2] = local_min.z;
            min_max[3] = local_max.x;
            min_max[4] = local_max.y;
            min_max[5] = local_max.z;
            block_num_x =
                    static_cast<uint32_t>(ceil((local_max.x - local_min.x) / (2 * seg_size * eb))) + 1;
            block_num_y =
                    static_cast<uint32_t>(ceil((local_max.y - local_min.y) / (2 * seg_size * eb))) + 1;

            // get lowest non-zero position, seg_size = 8 -> __ffs(8) = 4
            seg_shift = __ffs(seg_size) - 1;

        }
        block_num_x = __shfl_sync(0xffffffff, block_num_x, 0);
        block_num_y = __shfl_sync(0xffffffff, block_num_y, 0);
        seg_shift = __shfl_sync(0xffffffff, seg_shift, 0);


        // set mask, 0 -> 0b111
        uint32_t seg_mask = seg_size - 1;

#pragma unroll
        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
            uint64_t index = warp_base_start_idx + i * CHUNK_SIZE + tid;

            bool valid = index < coordinate_num;

            double x = valid ? x_coords[i] : 0.0f;
            double y = valid ? y_coords[i] : 0.0f;
            double z = valid ? z_coords[i] : 0.0f;

            // force to FFMA: a * b + c
            double quant_x_f = __fma_rn(x, inv_2eb, -local_min.x * inv_2eb);
            double quant_y_f = __fma_rn(y, inv_2eb, -local_min.y * inv_2eb);
            double quant_z_f = __fma_rn(z, inv_2eb, -local_min.z * inv_2eb);
            uint64_t quant_y = static_cast<uint64_t>(quant_y_f);
            uint64_t quant_x = static_cast<uint64_t>(quant_x_f);
            uint64_t quant_z = static_cast<uint64_t>(quant_z_f);

            // use shift as division ; & 0x7FFFFFFF remove highest sign bit
            uint64_t block_id = ((quant_x >> seg_shift) +
                                     ((quant_y >> seg_shift) * block_num_x) +
                                     ((quant_z >> seg_shift) * block_num_x * block_num_y)) & 0x7FFFFFFF;
            // use & seg_mask as modular
            uint64_t relative_pos = ((quant_x & seg_mask) +
                                         ((quant_y & seg_mask) * seg_size) +
                                         ((quant_z & seg_mask) * seg_size * seg_size)) & 0x7FFFFFFF;


            local_block_ids[i] = valid ? block_id : UINT64_MAX;
            local_rel_pos[i] = valid ? relative_pos : UINT64_MAX;


            int32_t block_id_bits = valid ? getEffectiveBits(block_id) : 1;
            int32_t rel_pos_bits = valid ? getEffectiveBits(relative_pos) : 1;

            local_block_max_block_id_bit = max(block_id_bits, local_block_max_block_id_bit);
            local_max_rel_pos_bit = max(rel_pos_bits, local_max_rel_pos_bit);
        }

        local_block_max_block_id_bit = warpReduceMax(local_block_max_block_id_bit);
        local_max_rel_pos_bit = warpReduceMax(local_max_rel_pos_bit);

        __syncthreads();

        if (warp_id == gridDim.x - 1) {
            BlockRadixSort(smem.sort_temp_storage).Sort(local_block_ids, local_rel_pos, 0,
                                                        local_block_max_block_id_bit + 1);
        } else {
            BlockRadixSort(smem.sort_temp_storage).Sort(local_block_ids, local_rel_pos, 0,
                                                        local_block_max_block_id_bit);
        }

        __syncthreads();

        uint32_t block_last_item_index =
                ELEMS_PER_BLOCK - ((warp_id == gridDim.x - 1) ? (gridDim.x * ELEMS_PER_BLOCK - coordinate_num) : 0) - 1;

        uint32_t thread_sum = 0;
        uint32_t thread_sum_prefix_sum = 0;
        uint64_t thread_last_item = local_block_ids[CHUNK_SIZE - 1];
        uint64_t pre_thread_last_item = __shfl_up_sync(0xffffffff, thread_last_item, 1);

#pragma unroll
        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
            uint32_t local_idx = tid * CHUNK_SIZE + i;
            uint64_t cur_val = local_block_ids[i];
            uint64_t prev_val = (i == 0) ? pre_thread_last_item : local_block_ids[i - 1];
            unsigned char tmp = (local_idx <= block_last_item_index) && (local_idx == 0 || cur_val != prev_val);
            sbuf[i][tid] = tmp;
            thread_sum += tmp;
            local_prefix_sum[i] = thread_sum;
        }
        thread_sum_prefix_sum = thread_sum;

#pragma unroll 5
        for (int32_t i = 1; i < BLOCK_SIZE; i <<= 1) {
            uint32_t tmp = __shfl_up_sync(0xffffffff, thread_sum_prefix_sum, i);
            if (tid >= i) thread_sum_prefix_sum += tmp;
        }

        uint32_t thread_prefix_offset = __shfl_up_sync(0xffffffff, thread_sum_prefix_sum, 1);
        int32_t add_offset = tid > 0;
#pragma unroll
        for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
            local_prefix_sum[i] += add_offset ? thread_prefix_offset : 0;
        }

        uint32_t unique_num = 0;
        if (tid == BLOCK_SIZE - 1) {
            unique_num = local_prefix_sum[CHUNK_SIZE - 1];
        }
        unique_num = __shfl_sync(0xffffffff, unique_num, BLOCK_SIZE - 1);

        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
            if (sbuf[i][tid]) {
                uint32_t boundary_pos = local_prefix_sum[i] - 1;
                smem.boundary_idx[boundary_pos] = tid * CHUNK_SIZE + i;
            }
        }
        __syncthreads();

        for (uint32_t k = tid; k < unique_num; k += BLOCK_SIZE) {
            uint32_t block_start_idx = smem.boundary_idx[k];
            uint32_t block_end_idx = (k < unique_num - 1) ? smem.boundary_idx[k + 1] :
                                         block_last_item_index + 1;
            uint32_t count = block_end_idx - block_start_idx;
            b_counts[k] = count;
            local_max_cnt_bit = max(getEffectiveBits(count), local_max_cnt_bit);
        }

        __syncthreads();

        uint64_t local_delta_bids[CHUNK_SIZE];
        uint64_t cur_thread_last_bid = local_block_ids[CHUNK_SIZE - 1];
        uint64_t pre_thread_last_bid = __shfl_up_sync(0xffffffff, cur_thread_last_bid, 1);
        uint64_t pre_local_bid = 0;
        int32_t local_delta_bid_count = 0;
#pragma unroll
        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
            if (sbuf[i][tid] == 0) continue;

            uint64_t b_id = local_block_ids[i];
            uint64_t delta = 0;

            if (local_delta_bid_count == 0) {
                delta = (tid == 0 && i == 0) ? 0 : b_id - pre_thread_last_bid;
            } else {
                delta = b_id - pre_local_bid;
            }

            pre_local_bid = b_id;
            local_delta_bids[local_delta_bid_count++] = delta;
            local_max_delta_block_id_bit = max(getEffectiveBits(delta), local_max_delta_block_id_bit);
        }


        uint32_t b_delta_bid_index_start = thread_sum_prefix_sum - thread_sum;

        for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
            if (i < local_delta_bid_count) {
                uint32_t index = b_delta_bid_index_start + i;
                // for each 32 write, padding：index + index / 32
                smem.b_delta_bids[index] = local_delta_bids[i];
            }
        }

        local_max_delta_block_id_bit = warpReduceMax(local_max_delta_block_id_bit);
        local_max_cnt_bit = warpReduceMax(local_max_cnt_bit);

        if (tid == 0) {
            max_bits_unique_num[0] = local_max_delta_block_id_bit;
            max_bits_unique_num[1] = local_max_cnt_bit;
            max_bits_unique_num[2] = local_max_rel_pos_bit;
            max_bits_unique_num[3] = unique_num;
            outlier = local_block_ids[0];
        }

        uint32_t b_delta_bids_bytes =
                (((((unique_num + 7) & ~7U) * local_max_delta_block_id_bit) >> 3) + 7) & ~7U;
        uint32_t b_count_bytes =
                (((((unique_num + 7) & ~7U) * local_max_cnt_bit) >> 3) + 7) & ~7U;
        uint32_t b_rel_pos_bytes = (((ELEMS_PER_BLOCK * local_max_rel_pos_bit) >> 3) + 7) & ~7U;
        uint64_t current_block_bytes = b_delta_bids_bytes + b_count_bytes + b_rel_pos_bytes;

        // save meta info
        unsigned char *meta_dst = comp_data + warp_id * META_DATA_BYTES;
#pragma unroll
        for (int32_t i = threadIdx.x; i < 6; i += 32) {
            reinterpret_cast<double *>(meta_dst)[i] = min_max[i];
        }

        if (threadIdx.x == 6) {
            reinterpret_cast<uint64_t *>(meta_dst)[6] = outlier;
        }

        if (threadIdx.x >= 7 && threadIdx.x <= 10) {
            reinterpret_cast<uint16_t *>(meta_dst + 56)[threadIdx.x - 7] = max_bits_unique_num[threadIdx.x - 7];
        }

        if (threadIdx.x == 11) {
            loc_offset[warp_id] = current_block_bytes;
        }

        __syncthreads();

        // write to fixed size memory region, use 3ULL to avoid overflow
        uint64_t base_idx = warp_id * ELEMS_PER_BLOCK * sizeof(uint64_t) * 3ULL + meta_offset;
        uint64_t base_cmp_byte_ofs = base_idx;
        uint64_t cmp_byte_ofs = base_cmp_byte_ofs;

        const uint32_t bytes_per_plane = (unique_num + 7) >> 3;
        for (int32_t bit = local_max_delta_block_id_bit - 1; bit >= 0; --bit) {
            // process each plane in each iteration, collect all bits in the same plane
            for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
                sbuf[i >> 5][i & 0x1F] = (smem.b_delta_bids[i] >> bit) & 0x1;
            }
            __syncthreads();

            // write the plane
            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                unsigned char byte = 0;
                const uint32_t start_bit = byte_idx * 8;

                for (int32_t b = 0; b < 8; ++b) {
                    const uint32_t pos = start_bit + b;
                    if (pos < unique_num) {
                        byte |= sbuf[pos >> 5][pos & 0x1F] << (7 - b);
                    }
                }
                const uint32_t plane_offset = (local_max_delta_block_id_bit - 1 - bit) * bytes_per_plane;
                comp_data[cmp_byte_ofs + plane_offset + byte_idx] = byte;
            }

            __syncthreads();
        }


        // save counts
        cmp_byte_ofs = base_cmp_byte_ofs + b_delta_bids_bytes;
        for (int32_t bit = local_max_cnt_bit - 1; bit >= 0; --bit) {

            for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
                // use b_rel_pos as a temporary buffer
                smem.b_rel_pos[i >> 5][i & 0x1F] = (b_counts[i] >> bit) & 0x1;
            }
            __syncthreads();

            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                unsigned char byte = 0;
                const uint32_t start_bit = byte_idx * 8;

                for (int32_t b = 0; b < 8; ++b) {
                    const uint32_t pos = start_bit + b;
                    if (pos < unique_num) {
                        byte |= smem.b_rel_pos[pos >> 5][pos & 0x1F] << (7 - b);
                    }
                }

                const uint32_t plane_offset = (local_max_cnt_bit - 1 - bit) * bytes_per_plane;
                comp_data[cmp_byte_ofs + plane_offset + byte_idx] = byte;
            }
            __syncthreads();
        }

        // save relative position
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            smem.b_rel_pos[tid][i] = local_rel_pos[i];
        }
        __syncthreads();

        cmp_byte_ofs =
                base_cmp_byte_ofs + b_delta_bids_bytes + b_count_bytes;
        for (int32_t bit = local_max_rel_pos_bit - 1; bit >= 0; --bit) {
#pragma unroll
            for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
                uint64_t val = smem.b_rel_pos[i][tid];
                uint32_t bit_val = (val >> bit) & 0x1;

                // __ballot_sync collect bit from 32 threads and packed to uint32_t
                uint32_t packed = __ballot_sync(0xFFFFFFFF, bit_val);

                if (tid == 0) {
                    int32_t plane_offset = (local_max_rel_pos_bit - 1 - bit) * (ELEMS_PER_BLOCK >> 3);
                    int32_t byte_idx = plane_offset + i * 4;

                    *reinterpret_cast<uint32_t *>(&comp_data[cmp_byte_ofs + byte_idx]) = packed;
                }
            }
        }
    }

    __global__ void
    runCompressKernel2D(const double *const __restrict__ d_x, const double *const __restrict__ d_y,
                      unsigned char *const __restrict__ comp_data,
                      volatile uint64_t *const __restrict__ loc_offset,
                      uint32_t seg_size, uint64_t coordinate_num, double eb) {
        __shared__ double min_max[6];
        __shared__ uint64_t outlier;

        using BlockRadixSort = cub::BlockRadixSort<uint64_t, BLOCK_SIZE, CHUNK_SIZE, uint64_t>;
        union SharedMemory {
            // Temporary storage for sorting
            typename BlockRadixSort::TempStorage sort_temp_storage;
            uint32_t boundary_idx[ELEMS_PER_BLOCK];
            uint64_t b_delta_bids[ELEMS_PER_BLOCK];
            uint64_t b_rel_pos[BLOCK_SIZE][CHUNK_SIZE + 1];
        };
        __shared__ SharedMemory smem;

        __shared__ uint16_t max_bits_unique_num[4];
        __shared__ uint16_t b_counts[ELEMS_PER_BLOCK];
        __shared__ unsigned char sbuf[CHUNK_SIZE][BLOCK_SIZE + 1];

        const uint64_t meta_offset = gridDim.x * META_DATA_BYTES;

        uint32_t tid = threadIdx.x;
        uint32_t bid = blockIdx.x;
        uint32_t bid_size = blockDim.x;
        uint32_t warp_id = (bid * bid_size + tid) >> 5;

        int32_t local_block_max_block_id_bit = -INT_MAX;
        int32_t local_max_delta_block_id_bit = -INT_MAX;
        int32_t local_max_cnt_bit = -INT_MAX;
        int32_t local_max_rel_pos_bit = -INT_MAX;

        uint64_t local_block_ids[CHUNK_SIZE];
        uint64_t local_rel_pos[CHUNK_SIZE];
        uint32_t local_prefix_sum[CHUNK_SIZE];

        // find min max within warp
        double3 local_min = make_double3(DBL_MAX, DBL_MAX, DBL_MAX);
        double3 local_max = make_double3(-DBL_MAX, -DBL_MAX, -DBL_MAX);

        double x_coords[CHUNK_SIZE];
        double y_coords[CHUNK_SIZE];
        uint64_t warp_base_start_idx = warp_id * ELEMS_PER_BLOCK;
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
            uint64_t index = warp_base_start_idx + i * BLOCK_SIZE + tid;

            bool valid = index < coordinate_num;

            double x = 0.0f, y = 0.0f;
            if (valid) {
                x = d_x[index];
                y = d_y[index];
            }
            x_coords[i] = x;
            y_coords[i] = y;

            // only valid points are used to calculate min/max
            if (valid) {
                local_min.x = fmin(local_min.x, x);
                local_min.y = fmin(local_min.y, y);

                local_max.x = fmax(local_max.x, x);
                local_max.y = fmax(local_max.y, y);
            }
        }

        // get min/max within warp and broadcast
        local_min.x = warpReduceMin(local_min.x);
        local_min.y = warpReduceMin(local_min.y);

        local_max.x = warpReduceMax(local_max.x);
        local_max.y = warpReduceMax(local_max.y);

        double inv_2eb = 1.0f / (2.0f * eb);
        uint64_t block_num_x = 0;
        uint32_t seg_shift = 0;
        if (tid == 0) {
            min_max[0] = local_min.x;
            min_max[1] = local_min.y;
            min_max[2] = 0;
            min_max[3] = local_max.x;
            min_max[4] = local_max.y;
            min_max[5] = 0;
            block_num_x =
                    static_cast<uint64_t>(ceil((local_max.x - local_min.x) / (2 * seg_size * eb))) + 1;
            // get lowest non-zero position, seg_size = 8 -> __ffs(8) = 4
            seg_shift = __ffs(seg_size) - 1;

        }
        block_num_x = __shfl_sync(0xffffffff, block_num_x, 0);
        seg_shift = __shfl_sync(0xffffffff, seg_shift, 0);

        // set mask, 0 -> 0b111
        uint32_t seg_mask = seg_size - 1;

#pragma unroll
        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
            uint64_t index = warp_base_start_idx + i * CHUNK_SIZE + tid;

            bool valid = index < coordinate_num;

            double x = valid ? x_coords[i] : 0.0f;
            double y = valid ? y_coords[i] : 0.0f;

            // force to FFMA: a * b + c
            double quant_x_f = __fma_rn(x, inv_2eb, -local_min.x * inv_2eb);
            double quant_y_f = __fma_rn(y, inv_2eb, -local_min.y * inv_2eb);
            uint64_t quant_y = static_cast<uint64_t>(quant_y_f);
            uint64_t quant_x = static_cast<uint64_t>(quant_x_f);

            // use shift as division ; & 0x7FFFFFFF remove highest sign bit
            uint64_t block_id = ((quant_x >> seg_shift) +
                                     ((quant_y >> seg_shift) * block_num_x)) & 0x7FFFFFFF;
            // use & seg_mask as modular
            uint64_t relative_pos = ((quant_x & seg_mask) +
                                         ((quant_y & seg_mask) * seg_size)) & 0x7FFFFFFF;


            local_block_ids[i] = valid ? block_id : UINT64_MAX;
            local_rel_pos[i] = valid ? relative_pos : UINT64_MAX;


            int32_t block_id_bits = valid ? getEffectiveBits(block_id) : 1;
            int32_t rel_pos_bits = valid ? getEffectiveBits(relative_pos) : 1;

            local_block_max_block_id_bit = max(block_id_bits, local_block_max_block_id_bit);
            local_max_rel_pos_bit = max(rel_pos_bits, local_max_rel_pos_bit);
        }

        local_block_max_block_id_bit = warpReduceMax(local_block_max_block_id_bit);
        local_max_rel_pos_bit = warpReduceMax(local_max_rel_pos_bit);

        __syncthreads();

        if (warp_id == gridDim.x - 1) {
            BlockRadixSort(smem.sort_temp_storage).Sort(local_block_ids, local_rel_pos, 0,
                                                        local_block_max_block_id_bit + 1);
        } else {
            BlockRadixSort(smem.sort_temp_storage).Sort(local_block_ids, local_rel_pos, 0,
                                                        local_block_max_block_id_bit);
        }

        __syncthreads();

        uint32_t block_last_item_index =
                ELEMS_PER_BLOCK - ((warp_id == gridDim.x - 1) ? (gridDim.x * ELEMS_PER_BLOCK - coordinate_num) : 0) - 1;

        uint32_t thread_sum = 0;
        uint32_t thread_sum_prefix_sum = 0;
        uint64_t thread_last_item = local_block_ids[CHUNK_SIZE - 1];
        uint64_t pre_thread_last_item = __shfl_up_sync(0xffffffff, thread_last_item, 1);

#pragma unroll
        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
            uint32_t local_idx = tid * CHUNK_SIZE + i;
            uint64_t cur_val = local_block_ids[i];
            uint64_t prev_val = (i == 0) ? pre_thread_last_item : local_block_ids[i - 1];
            unsigned char tmp = (local_idx <= block_last_item_index) && (local_idx == 0 || cur_val != prev_val);
            sbuf[i][tid] = tmp;
            thread_sum += tmp;
            local_prefix_sum[i] = thread_sum;
        }
        thread_sum_prefix_sum = thread_sum;

#pragma unroll 5
        for (int32_t i = 1; i < BLOCK_SIZE; i <<= 1) {
            uint32_t tmp = __shfl_up_sync(0xffffffff, thread_sum_prefix_sum, i);
            if (tid >= i) thread_sum_prefix_sum += tmp;
        }

        uint32_t thread_prefix_offset = __shfl_up_sync(0xffffffff, thread_sum_prefix_sum, 1);
        int32_t add_offset = tid > 0;
#pragma unroll
        for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
            local_prefix_sum[i] += add_offset ? thread_prefix_offset : 0;
        }

        uint32_t unique_num = 0;
        if (tid == BLOCK_SIZE - 1) {
            unique_num = local_prefix_sum[CHUNK_SIZE - 1];
        }
        unique_num = __shfl_sync(0xffffffff, unique_num, BLOCK_SIZE - 1);

        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
            if (sbuf[i][tid]) {
                uint32_t boundary_pos = local_prefix_sum[i] - 1;
                smem.boundary_idx[boundary_pos] = tid * CHUNK_SIZE + i;
            }
        }
        __syncthreads();

        for (uint32_t k = tid; k < unique_num; k += BLOCK_SIZE) {
            uint32_t block_start_idx = smem.boundary_idx[k];
            uint32_t block_end_idx = (k < unique_num - 1) ? smem.boundary_idx[k + 1] :
                                         block_last_item_index + 1;
            uint32_t count = block_end_idx - block_start_idx;
            b_counts[k] = count;
            local_max_cnt_bit = max(getEffectiveBits(count), local_max_cnt_bit);
        }


        uint64_t local_delta_bids[CHUNK_SIZE];
        uint64_t cur_thread_last_bid = local_block_ids[CHUNK_SIZE - 1];
        uint64_t pre_thread_last_bid = __shfl_up_sync(0xffffffff, cur_thread_last_bid, 1);
        uint64_t pre_local_bid = 0;
        int32_t local_delta_bid_count = 0;
#pragma unroll
        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
            if (sbuf[i][tid] == 0) continue;

            uint64_t b_id = local_block_ids[i];
            uint64_t delta = 0;

            if (local_delta_bid_count == 0) {
                delta = (tid == 0 && i == 0) ? 0 : b_id - pre_thread_last_bid;
            } else {
                delta = b_id - pre_local_bid;
            }

            pre_local_bid = b_id;
            local_delta_bids[local_delta_bid_count++] = delta;
            local_max_delta_block_id_bit = max(getEffectiveBits(delta), local_max_delta_block_id_bit);
        }


        uint32_t b_delta_bid_index_start = thread_sum_prefix_sum - thread_sum;

        for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
            if (i < local_delta_bid_count) {
                uint32_t index = b_delta_bid_index_start + i;
                // for each 32 write, padding：index + index / 32
                smem.b_delta_bids[index] = local_delta_bids[i];
            }
        }

        local_max_delta_block_id_bit = warpReduceMax(local_max_delta_block_id_bit);
        local_max_cnt_bit = warpReduceMax(local_max_cnt_bit);

        if (tid == 0) {
            max_bits_unique_num[0] = local_max_delta_block_id_bit;
            max_bits_unique_num[1] = local_max_cnt_bit;
            max_bits_unique_num[2] = local_max_rel_pos_bit;
            max_bits_unique_num[3] = unique_num;
            outlier = local_block_ids[0];
        }

        uint32_t b_delta_bids_bytes =
                (((((unique_num + 7) & ~7U) * local_max_delta_block_id_bit) >> 3) + 7) & ~7U;
        uint32_t b_count_bytes =
                (((((unique_num + 7) & ~7U) * local_max_cnt_bit) >> 3) + 3) & ~3U;
        uint32_t b_rel_pos_bytes = (((ELEMS_PER_BLOCK * local_max_rel_pos_bit) >> 3) + 7) & ~7U;
        uint64_t current_block_bytes = b_delta_bids_bytes + b_count_bytes + b_rel_pos_bytes;

        // save meta info
        unsigned char *meta_dst = comp_data + warp_id * META_DATA_BYTES;
#pragma unroll
        for (int32_t i = threadIdx.x; i < 6; i += 32) {
            reinterpret_cast<double *>(meta_dst)[i] = min_max[i];
        }

        if (threadIdx.x == 6) {
            reinterpret_cast<uint64_t *>(meta_dst)[6] = outlier;
        }

        if (threadIdx.x >= 7 && threadIdx.x <= 10) {
            reinterpret_cast<uint16_t *>(meta_dst + 56)[threadIdx.x - 7] = max_bits_unique_num[threadIdx.x - 7];
        }

        if (threadIdx.x == 11) {
            loc_offset[warp_id] = current_block_bytes;
        }

        __syncthreads();

        // write to fixed size memory region, use 3ULL to avoid overflow
        uint64_t base_idx = warp_id * ELEMS_PER_BLOCK * sizeof(uint64_t) * 3ULL + meta_offset;
        uint64_t base_cmp_byte_ofs = base_idx;
        uint64_t cmp_byte_ofs = base_cmp_byte_ofs;

        const uint32_t bytes_per_plane = (unique_num + 7) >> 3;
        for (int32_t bit = local_max_delta_block_id_bit - 1; bit >= 0; --bit) {
            // process each plane in each iteration, collect all bits in the same plane
            for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
                sbuf[i >> 5][i & 0x1F] = (smem.b_delta_bids[i] >> bit) & 0x1;
            }
            __syncthreads();

            // write the plane
            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                unsigned char byte = 0;
                const uint32_t start_bit = byte_idx * 8;

                for (int32_t b = 0; b < 8; ++b) {
                    const uint32_t pos = start_bit + b;
                    if (pos < unique_num) {
                        byte |= sbuf[pos >> 5][pos & 0x1F] << (7 - b);
                    }
                }
                const uint32_t plane_offset = (local_max_delta_block_id_bit - 1 - bit) * bytes_per_plane;
                comp_data[cmp_byte_ofs + plane_offset + byte_idx] = byte;
            }

            __syncthreads();
        }

        // save counts
        cmp_byte_ofs = base_cmp_byte_ofs + b_delta_bids_bytes;
        for (int32_t bit = local_max_cnt_bit - 1; bit >= 0; --bit) {

            for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
                // use b_rel_pos as a temporary buffer
                smem.b_rel_pos[i >> 5][i & 0x1F] = (b_counts[i] >> bit) & 0x1;
            }
            __syncthreads();

            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                unsigned char byte = 0;
                const uint32_t start_bit = byte_idx * 8;

                for (int32_t b = 0; b < 8; ++b) {
                    const uint32_t pos = start_bit + b;
                    if (pos < unique_num) {
                        byte |= smem.b_rel_pos[pos >> 5][pos & 0x1F] << (7 - b);
                    }
                }

                const uint32_t plane_offset = (local_max_cnt_bit - 1 - bit) * bytes_per_plane;
                comp_data[cmp_byte_ofs + plane_offset + byte_idx] = byte;
            }
            __syncthreads();
        }

        // save relative position
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            smem.b_rel_pos[tid][i] = local_rel_pos[i];
        }
        __syncthreads();

        cmp_byte_ofs =
                base_cmp_byte_ofs + b_delta_bids_bytes + b_count_bytes;
        for (int32_t bit = local_max_rel_pos_bit - 1; bit >= 0; --bit) {
#pragma unroll
            for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
                uint64_t val = smem.b_rel_pos[i][tid];
                uint32_t bit_val = (val >> bit) & 0x1;

                // __ballot_sync collect bit from 32 threads and packed to uint32_t
                uint32_t packed = __ballot_sync(0xFFFFFFFF, bit_val);

                if (tid == 0) {
                    int32_t plane_offset = (local_max_rel_pos_bit - 1 - bit) * (ELEMS_PER_BLOCK >> 3);
                    int32_t byte_idx = plane_offset + i * 4;

                    *reinterpret_cast<uint32_t *>(&comp_data[cmp_byte_ofs + byte_idx]) = packed;
                }
            }
        }
    }

    __global__ void
    runDecompressKernel(double *decData_x, double *decData_y, double *decData_z,
                        unsigned char *const __restrict__ comp_data, const uint64_t *const __restrict__ comp_offset,
                        uint32_t seg_size, uint64_t coordinate_num, double eb) {
        __shared__ double min_max[6];
        __shared__ uint64_t outlier;
//        __shared__ uint32_t d_xyz_block_num[3];

        __shared__ uint64_t b_buff_block_id[BLOCK_SIZE * CHUNK_SIZE];
        __shared__ uint64_t b_buff_counts_pos[BLOCK_SIZE * CHUNK_SIZE];

        __shared__ uint16_t max_bits_unique_num[4];


        uint32_t tid = threadIdx.x;
        uint32_t bid = blockIdx.x;
        uint32_t bid_size = blockDim.x;
        uint32_t glb_tid = bid * bid_size + tid;
        uint32_t warp_id = glb_tid >> 5;
        uint64_t block_num_x, block_num_y;
        if (tid == 0) {
            const unsigned char *meta_src = comp_data + warp_id * META_DATA_BYTES;
            const double *f_src = reinterpret_cast<const double *>(meta_src);
            for (int32_t i = 0; i < 6; ++i) {
                min_max[i] = f_src[i];
            }

            outlier = reinterpret_cast<const uint64_t *>(meta_src)[6];

            const uint16_t *u16_src = reinterpret_cast<const uint16_t *>(meta_src + 56);
            for (int32_t i = 0; i < 4; ++i) {
                max_bits_unique_num[i] = u16_src[i];
            }

            block_num_x =
                    static_cast<uint32_t>(ceil((min_max[3] - min_max[0]) / (2 * seg_size * eb))) + 1;
            block_num_y =
                    static_cast<uint32_t>(ceil((min_max[4] - min_max[1]) / (2 * seg_size * eb))) + 1;
        }
        block_num_x = __shfl_sync(0xffffffff, block_num_x, 0);
        block_num_y = __shfl_sync(0xffffffff, block_num_y, 0);

        __syncthreads();

        uint32_t b_delta_bids_bytes =
                (((((max_bits_unique_num[3] + 7) & ~7U) * max_bits_unique_num[0]) >> 3) + 7) & ~7U;
        uint32_t b_count_bytes =
                (((((max_bits_unique_num[3] + 7) & ~7U) * max_bits_unique_num[1]) >> 3) + 7) & ~7U;
        uint64_t meta_offset = gridDim.x * META_DATA_BYTES;
        uint64_t base_cmp_byte_ofs = comp_offset[warp_id] + meta_offset;
        uint64_t cmp_byte_ofs = base_cmp_byte_ofs;

        const uint16_t unique_num = max_bits_unique_num[3];
        const uint16_t bytes_per_plane = (unique_num + 7) >> 3;
        uint16_t total_bits = max_bits_unique_num[0];

        const uint16_t full_bytes = unique_num >> 3;
        const uint16_t remaining_bits = unique_num & 0x1F;

        for (uint32_t i = tid; i < ELEMS_PER_BLOCK; i += BLOCK_SIZE) {
            b_buff_block_id[i] = 0;
            b_buff_counts_pos[i] = 0;
        }

        __syncthreads();


        for (uint32_t plane = 0; plane < total_bits; ++plane) {
            const uint32_t src_bit_pos = total_bits - 1 - plane;
            const uint32_t plane_offset = plane * bytes_per_plane;

            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                const unsigned char byte = comp_data[cmp_byte_ofs + plane_offset + byte_idx];
                const uint32_t base_pos = byte_idx * 8;

                uint32_t valid_bits = (byte_idx < full_bytes) ? 8 : remaining_bits;

#pragma unroll
                for (int32_t b = 0; b < 8; ++b) {
                    if (b < valid_bits) {
                        const uint32_t pos = base_pos + b;
                        const unsigned char bit = (byte >> (7 - b)) & 0x1;
                        b_buff_block_id[pos] |= (bit << src_bit_pos);
                    }
                }
            }
            __syncthreads();
        }

        cmp_byte_ofs = base_cmp_byte_ofs + b_delta_bids_bytes;
        total_bits = max_bits_unique_num[1];

        for (uint32_t plane = 0; plane < total_bits; ++plane) {
            const uint32_t src_bit_pos = total_bits - 1 - plane;
            const uint32_t plane_offset = plane * bytes_per_plane;

            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                const unsigned char byte = comp_data[cmp_byte_ofs + plane_offset + byte_idx];
                const uint32_t base_pos = byte_idx * 8;

                uint32_t valid_bits = (byte_idx < full_bytes) ? 8 : remaining_bits;

#pragma unroll
                for (int32_t b = 0; b < 8; ++b) {
                    if (b < valid_bits) {
                        const uint32_t pos = base_pos + b;
                        const unsigned char bit = (byte >> (7 - b)) & 0x1;
                        b_buff_counts_pos[pos] |= (bit << src_bit_pos);
                    }
                }
            }
            __syncthreads();
        }

        total_bits = max_bits_unique_num[2];
        cmp_byte_ofs =
                base_cmp_byte_ofs + b_delta_bids_bytes + b_count_bytes;

        uint64_t val[CHUNK_SIZE] = {0};
        for (int32_t bit = total_bits - 1; bit >= 0; --bit) {
            int32_t plane_offset = (total_bits - 1 - bit) * (ELEMS_PER_BLOCK >> 3);

#pragma unroll
            for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
                int32_t byte_idx = plane_offset + i * 4;
                uint32_t packed = *reinterpret_cast<const uint32_t *>(&comp_data[cmp_byte_ofs + byte_idx]);

                uint32_t bit_val = (packed >> tid) & 0x1;
                val[i] |= (bit_val << bit);
            }
        }

        if (tid == 0) {
            b_buff_block_id[0] = outlier;
            for (int32_t i = 1; i <= unique_num; ++i) {
                b_buff_block_id[i] = b_buff_block_id[i] + b_buff_block_id[i - 1];
            }
            for (int32_t i = 1; i < unique_num; ++i) {
                b_buff_counts_pos[i] = b_buff_counts_pos[i] + b_buff_counts_pos[i - 1];
            }
        }
        __syncthreads();


        uint16_t local_counts[CHUNK_SIZE];
        uint64_t local_buffs[CHUNK_SIZE];

        int32_t local_idx = 0;
        for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
            local_counts[local_idx] = b_buff_counts_pos[i];
            local_buffs[local_idx] = b_buff_block_id[i];
            ++local_idx;
        }
        __syncthreads();


        local_idx = 0;
        for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
            uint16_t cur = local_counts[local_idx];
            uint16_t prev = (i == 0) ? 0 : b_buff_counts_pos[i - 1];
            for (int32_t j = cur - 1; j >= prev; --j) {
                b_buff_block_id[j] = local_buffs[local_idx];
            }
            ++local_idx;
        }

        __syncthreads();

        for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
            b_buff_counts_pos[i * CHUNK_SIZE + tid] = val[i];
        }
        __syncthreads();

        uint64_t global_block_start = bid * BLOCK_SIZE * CHUNK_SIZE;
        uint64_t total_valid_points = min(coordinate_num - global_block_start, (uint64_t) (BLOCK_SIZE * CHUNK_SIZE));

        // SOA input
        double *x_ptr = decData_x + global_block_start;
        double *y_ptr = decData_y + global_block_start;
        double *z_ptr = decData_z + global_block_start;

        int32_t seg_shift = __ffs(seg_size) - 1;
        uint32_t seg_mask = seg_size - 1;
        for (uint64_t i = tid; i < total_valid_points; i += BLOCK_SIZE) {
            uint64_t rel_pos = b_buff_counts_pos[i];
            uint64_t blk_id = b_buff_block_id[i];

            uint64_t block_x = block_num_x;
            uint64_t block_y = block_num_y;

            uint64_t bx = blk_id % block_x;
            uint64_t by = (blk_id / block_x) % block_y;
            uint64_t bz = blk_id / (block_x * block_y);

            uint64_t rx = rel_pos & seg_mask;
            uint64_t ry = (rel_pos >> seg_shift) & seg_mask;
            uint64_t rz = rel_pos >> (seg_shift * 2);

            uint64_t qx = bx * seg_size + rx;
            uint64_t qy = by * seg_size + ry;
            uint64_t qz = bz * seg_size + rz;

            double x = min_max[0] + ((qx << 1) | 1) * eb;
            double y = min_max[1] + ((qy << 1) | 1) * eb;
            double z = min_max[2] + ((qz << 1) | 1) * eb;

            x_ptr[i] = x;
            y_ptr[i] = y;
            z_ptr[i] = z;
        }
    }

    __global__ void
    runDecompressKernel2D(double *decData_x, double *decData_y,
                        unsigned char *const __restrict__ comp_data, const uint64_t *const __restrict__ comp_offset,
                        uint32_t seg_size, uint64_t coordinate_num, double eb) {
        __shared__ double min_max[6];
        __shared__ uint64_t outlier;
//        __shared__ uint32_t d_xyz_block_num[3];

        __shared__ uint64_t b_buff_block_id[BLOCK_SIZE * CHUNK_SIZE];
        __shared__ uint64_t b_buff_counts_pos[BLOCK_SIZE * CHUNK_SIZE];
        __shared__ uint16_t max_bits_unique_num[4];


        uint32_t tid = threadIdx.x;
        uint32_t bid = blockIdx.x;
        uint32_t bid_size = blockDim.x;
        uint32_t glb_tid = bid * bid_size + tid;
        uint32_t warp_id = glb_tid >> 5;
        uint64_t block_num_x, block_num_y;
        if (tid == 0) {
            const unsigned char *meta_src = comp_data + warp_id * META_DATA_BYTES;
            const double *f_src = reinterpret_cast<const double *>(meta_src);
            for (int32_t i = 0; i < 6; ++i) {
                min_max[i] = f_src[i];
            }

            outlier = reinterpret_cast<const uint64_t *>(meta_src)[6];

            const uint16_t *u16_src = reinterpret_cast<const uint16_t *>(meta_src + 56);
            for (int32_t i = 0; i < 4; ++i) {
                max_bits_unique_num[i] = u16_src[i];
            }

            block_num_x =
                    static_cast<uint32_t>(ceil((min_max[3] - min_max[0]) / (2 * seg_size * eb))) + 1;
            block_num_y =
                    static_cast<uint32_t>(ceil((min_max[4] - min_max[1]) / (2 * seg_size * eb))) + 1;
        }
        block_num_x = __shfl_sync(0xffffffff, block_num_x, 0);
        block_num_y = __shfl_sync(0xffffffff, block_num_y, 0);

        __syncthreads();

        uint32_t b_delta_bids_bytes =
                (((((max_bits_unique_num[3] + 7) & ~7U) * max_bits_unique_num[0]) >> 3) + 7) & ~7U;
        uint32_t b_count_bytes =
                (((((max_bits_unique_num[3] + 7) & ~7U) * max_bits_unique_num[1]) >> 3) + 7) & ~7U;
        uint64_t meta_offset = gridDim.x * META_DATA_BYTES;
        uint64_t base_cmp_byte_ofs = comp_offset[warp_id] + meta_offset;
        uint64_t cmp_byte_ofs = base_cmp_byte_ofs;

        const uint16_t unique_num = max_bits_unique_num[3];
        const uint16_t bytes_per_plane = (unique_num + 7) >> 3;
        uint16_t total_bits = max_bits_unique_num[0];

        const uint16_t full_bytes = unique_num >> 3;
        const uint16_t remaining_bits = unique_num & 0x1F;

        for (uint32_t i = tid; i < ELEMS_PER_BLOCK; i += BLOCK_SIZE) {
            b_buff_block_id[i] = 0;
            b_buff_counts_pos[i] = 0;
        }

        __syncthreads();


        for (uint32_t plane = 0; plane < total_bits; ++plane) {
            const uint32_t src_bit_pos = total_bits - 1 - plane;
            const uint32_t plane_offset = plane * bytes_per_plane;

            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                const unsigned char byte = comp_data[cmp_byte_ofs + plane_offset + byte_idx];
                const uint32_t base_pos = byte_idx * 8;

                uint32_t valid_bits = (byte_idx < full_bytes) ? 8 : remaining_bits;

#pragma unroll
                for (int32_t b = 0; b < 8; ++b) {
                    if (b < valid_bits) {
                        const uint32_t pos = base_pos + b;
                        const unsigned char bit = (byte >> (7 - b)) & 0x1;
                        b_buff_block_id[pos] |= (bit << src_bit_pos);
                    }
                }
            }
            __syncthreads();
        }

        cmp_byte_ofs = base_cmp_byte_ofs + b_delta_bids_bytes;
        total_bits = max_bits_unique_num[1];

        for (uint32_t plane = 0; plane < total_bits; ++plane) {
            const uint32_t src_bit_pos = total_bits - 1 - plane;
            const uint32_t plane_offset = plane * bytes_per_plane;

            for (uint32_t byte_idx = tid; byte_idx < bytes_per_plane; byte_idx += BLOCK_SIZE) {
                const unsigned char byte = comp_data[cmp_byte_ofs + plane_offset + byte_idx];
                const uint32_t base_pos = byte_idx * 8;

                uint32_t valid_bits = (byte_idx < full_bytes) ? 8 : remaining_bits;

#pragma unroll
                for (int32_t b = 0; b < 8; ++b) {
                    if (b < valid_bits) {
                        const uint32_t pos = base_pos + b;
                        const unsigned char bit = (byte >> (7 - b)) & 0x1;
                        b_buff_counts_pos[pos] |= (bit << src_bit_pos);
                    }
                }
            }
            __syncthreads();
        }

        total_bits = max_bits_unique_num[2];
        cmp_byte_ofs =
                base_cmp_byte_ofs + b_delta_bids_bytes + b_count_bytes;

        uint64_t val[CHUNK_SIZE] = {0};
        for (int32_t bit = total_bits - 1; bit >= 0; --bit) {
            int32_t plane_offset = (total_bits - 1 - bit) * (ELEMS_PER_BLOCK >> 3);

#pragma unroll
            for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
                int32_t byte_idx = plane_offset + i * 4;
                uint32_t packed = *reinterpret_cast<const uint32_t *>(&comp_data[cmp_byte_ofs + byte_idx]);

                uint32_t bit_val = (packed >> tid) & 0x1;
                val[i] |= (bit_val << bit);
            }
        }


        if (tid == 0) {
            b_buff_block_id[0] = outlier;
            for (int32_t i = 1; i <= unique_num; ++i) {
                b_buff_block_id[i] = b_buff_block_id[i] + b_buff_block_id[i - 1];
            }
            for (int32_t i = 1; i < unique_num; ++i) {
                b_buff_counts_pos[i] = b_buff_counts_pos[i] + b_buff_counts_pos[i - 1];
            }
        }
        __syncthreads();


        uint16_t local_counts[CHUNK_SIZE];
        uint64_t local_buffs[CHUNK_SIZE];

        int32_t local_idx = 0;
        for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
            local_counts[local_idx] = b_buff_counts_pos[i];
            local_buffs[local_idx] = b_buff_block_id[i];
            ++local_idx;
        }
        __syncthreads();


        local_idx = 0;
        for (uint32_t i = tid; i < unique_num; i += BLOCK_SIZE) {
            uint16_t cur = local_counts[local_idx];
            uint16_t prev = (i == 0) ? 0 : b_buff_counts_pos[i - 1];
            for (int32_t j = cur - 1; j >= prev; --j) {
                b_buff_block_id[j] = local_buffs[local_idx];
            }
            ++local_idx;
        }

        __syncthreads();

        for (int32_t i = 0; i < CHUNK_SIZE; ++i) {
            b_buff_counts_pos[i * CHUNK_SIZE + tid] = val[i];
        }
        __syncthreads();

        uint64_t global_block_start = bid * BLOCK_SIZE * CHUNK_SIZE;
        uint64_t total_valid_points = min(coordinate_num - global_block_start, (uint64_t) (BLOCK_SIZE * CHUNK_SIZE));

        // SOA input
        double *x_ptr = decData_x + global_block_start;
        double *y_ptr = decData_y + global_block_start;

        int32_t seg_shift = __ffs(seg_size) - 1;
        uint32_t seg_mask = seg_size - 1;
        for (uint64_t i = tid; i < total_valid_points; i += BLOCK_SIZE) {
            uint64_t rel_pos = b_buff_counts_pos[i];
            uint64_t blk_id = b_buff_block_id[i];

            uint64_t block_x = block_num_x;
            uint64_t block_y = block_num_y;

            uint64_t bx = blk_id % block_x;
            uint64_t by = (blk_id / block_x) % block_y;

            uint64_t rx = rel_pos & seg_mask;
            uint64_t ry = (rel_pos >> seg_shift) & seg_mask;

            uint64_t qx = bx * seg_size + rx;
            uint64_t qy = by * seg_size + ry;

            double x = min_max[0] + ((qx << 1) | 1) * eb;
            double y = min_max[1] + ((qy << 1) | 1) * eb;

            x_ptr[i] = x;
            y_ptr[i] = y;
        }
    }

    __global__ void sampleMinMaxXYZKernel(const double *d_x, const double *d_y, const double *d_z,
                                          double *d_min_x, double *d_min_y, double *d_min_z, double *d_max_x, double *d_max_y,
                                          double *d_max_z, int32_t sample_length, uint64_t sample_stride, uint64_t N) {
        extern __shared__ double sdata[];
        double *smin_x = sdata;
        double *smax_x = &smin_x[sample_length];
        double *smin_y = &smax_x[sample_length];
        double *smax_y = &smin_y[sample_length];
        double *smin_z = &smax_y[sample_length];
        double *smax_z = &smin_z[sample_length];

        int32_t tid = threadIdx.x;
        uint64_t block_start = blockIdx.x * sample_stride;

        if (block_start + sample_length > N) return;

        if (tid < sample_length) {
            uint64_t idx = block_start + tid;
            double val_x = d_x[idx];
            double val_y = d_y[idx];
            double val_z = d_z[idx];

            smin_x[tid] = val_x;
            smax_x[tid] = val_x;
            smin_y[tid] = val_y;
            smax_y[tid] = val_y;
            smin_z[tid] = val_z;
            smax_z[tid] = val_z;
        }

        __syncthreads();

        // reduction for min/max
#pragma unroll
        for (int32_t stride = sample_length / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smin_x[tid] = fmin(smin_x[tid], smin_x[tid + stride]);
                smax_x[tid] = fmax(smax_x[tid], smax_x[tid + stride]);

                smin_y[tid] = fmin(smin_y[tid], smin_y[tid + stride]);
                smax_y[tid] = fmax(smax_y[tid], smax_y[tid + stride]);

                smin_z[tid] = fmin(smin_z[tid], smin_z[tid + stride]);
                smax_z[tid] = fmax(smax_z[tid], smax_z[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_min_x[blockIdx.x] = smin_x[0];
            d_min_y[blockIdx.x] = smin_y[0];
            d_min_z[blockIdx.x] = smin_z[0];

            d_max_x[blockIdx.x] = smax_x[0];
            d_max_y[blockIdx.x] = smax_y[0];
            d_max_z[blockIdx.x] = smax_z[0];
        }
    }

    __global__ void sampleMinMaxXYZKernel2D(const double *d_x, const double *d_y,
                                            double *d_min_x, double *d_min_y, double *d_min_z, double *d_max_x, double *d_max_y,
                                            double *d_max_z, int32_t sample_length, uint64_t sample_stride, uint64_t N) {
        extern __shared__ double sdata[];
        double *smin_x = sdata;
        double *smax_x = &smin_x[sample_length];
        double *smin_y = &smax_x[sample_length];
        double *smax_y = &smin_y[sample_length];

        int32_t tid = threadIdx.x;
        uint64_t block_start = blockIdx.x * sample_stride;

        if (block_start + sample_length > N) return;

        if (tid < sample_length) {
            uint64_t idx = block_start + tid;
            double val_x = d_x[idx];
            double val_y = d_y[idx];

            smin_x[tid] = val_x;
            smax_x[tid] = val_x;
            smin_y[tid] = val_y;
            smax_y[tid] = val_y;
        }

        __syncthreads();

        // reduction for min/max
#pragma unroll
        for (int32_t stride = sample_length / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smin_x[tid] = fmin(smin_x[tid], smin_x[tid + stride]);
                smax_x[tid] = fmax(smax_x[tid], smax_x[tid + stride]);

                smin_y[tid] = fmin(smin_y[tid], smin_y[tid + stride]);
                smax_y[tid] = fmax(smax_y[tid], smax_y[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_min_x[blockIdx.x] = smin_x[0];
            d_min_y[blockIdx.x] = smin_y[0];

            d_max_x[blockIdx.x] = smax_x[0];
            d_max_y[blockIdx.x] = smax_y[0];
        }
    }


    __global__ void reduceMinMax(const double *min_x, const double *max_x,
                                 const double *min_y, const double *max_y,
                                 const double *min_z, const double *max_z,
                                 double *out_block_results,
                                 int32_t N) {
        extern __shared__ double sdata[];
        double *s_min_x = sdata + 0 * blockDim.x;
        double *s_max_x = sdata + 1 * blockDim.x;
        double *s_min_y = sdata + 2 * blockDim.x;
        double *s_max_y = sdata + 3 * blockDim.x;
        double *s_min_z = sdata + 4 * blockDim.x;
        double *s_max_z = sdata + 5 * blockDim.x;

        int32_t tid = threadIdx.x;
        int32_t i = blockIdx.x * blockDim.x + tid;

        s_min_x[tid] = s_min_y[tid] = s_min_z[tid] = DBL_MAX;
        s_max_x[tid] = s_max_y[tid] = s_max_z[tid] = DBL_MIN;

        if (i < N) {
            s_min_x[tid] = min_x[i];
            s_max_x[tid] = max_x[i];
            s_min_y[tid] = min_y[i];
            s_max_y[tid] = max_y[i];
            s_min_z[tid] = min_z[i];
            s_max_z[tid] = max_z[i];
        }
        __syncthreads();

        // Block reduction
        for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_min_x[tid] = fmin(s_min_x[tid], s_min_x[tid + stride]);
                s_max_x[tid] = fmax(s_max_x[tid], s_max_x[tid + stride]);
                s_min_y[tid] = fmin(s_min_y[tid], s_min_y[tid + stride]);
                s_max_y[tid] = fmax(s_max_y[tid], s_max_y[tid + stride]);
                s_min_z[tid] = fmin(s_min_z[tid], s_min_z[tid + stride]);
                s_max_z[tid] = fmax(s_max_z[tid], s_max_z[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicMinDbl(&out_block_results[0], s_min_x[0]);
            atomicMaxDbl(&out_block_results[1], s_max_x[0]);
            atomicMinDbl(&out_block_results[2], s_min_y[0]);
            atomicMaxDbl(&out_block_results[3], s_max_y[0]);
            atomicMinDbl(&out_block_results[4], s_min_z[0]);
            atomicMaxDbl(&out_block_results[5], s_max_z[0]);
        }
    }

    __global__ void reduceMinMax2D(const double *min_x, const double *max_x,
                                   const double *min_y, const double *max_y,
                                   double *out_block_results,
                                   int32_t N) {
        extern __shared__ double sdata[];
        double *s_min_x = sdata + 0 * blockDim.x;
        double *s_max_x = sdata + 1 * blockDim.x;
        double *s_min_y = sdata + 2 * blockDim.x;
        double *s_max_y = sdata + 3 * blockDim.x;

        int32_t tid = threadIdx.x;
        int32_t i = blockIdx.x * blockDim.x + tid;

        s_min_x[tid] = s_min_y[tid] = DBL_MAX;
        s_max_x[tid] = s_max_y[tid] = DBL_MIN;

        if (i < N) {
            s_min_x[tid] = min_x[i];
            s_max_x[tid] = max_x[i];
            s_min_y[tid] = min_y[i];
            s_max_y[tid] = max_y[i];
        }
        __syncthreads();

        // Block reduction
        for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_min_x[tid] = fmin(s_min_x[tid], s_min_x[tid + stride]);
                s_max_x[tid] = fmax(s_max_x[tid], s_max_x[tid + stride]);
                s_min_y[tid] = fmin(s_min_y[tid], s_min_y[tid + stride]);
                s_max_y[tid] = fmax(s_max_y[tid], s_max_y[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicMinDbl(&out_block_results[0], s_min_x[0]);
            atomicMaxDbl(&out_block_results[1], s_max_x[0]);
            atomicMinDbl(&out_block_results[2], s_min_y[0]);
            atomicMaxDbl(&out_block_results[3], s_max_y[0]);
        }
    }
}



void CudaWarpCompressDouble::computeSegSize(double *d_x, double *d_y, double *d_z, double *d_cache, CudaConfig &cuda_config) {
    int32_t sample_length = 256;
    int32_t num_blocks = 1024;

    uint64_t stride_between_samples = max((cuda_config.coordinate_num - sample_length) / num_blocks, 1UL);
    uint32_t shared_mem_bytes = sample_length * 6 * sizeof(double);

    double *d_min_x = d_cache;
    double *d_min_y = d_cache + num_blocks;
    double *d_min_z = d_cache + num_blocks * 2;

    double *d_max_x = d_cache + num_blocks * 3;
    double *d_max_y = d_cache + num_blocks * 4;
    double *d_max_z = d_cache + num_blocks * 5;

    double *d_output_minmax;
    double init_vals[6] = {
            DBL_MAX, -DBL_MAX,
            DBL_MAX, -DBL_MAX,
            DBL_MAX, -DBL_MAX
    };
    cudaMalloc(&d_output_minmax, sizeof(double) * 6);
    cudaMemcpy(d_output_minmax, init_vals, sizeof(init_vals), cudaMemcpyHostToDevice);

    int32_t block_reduce_num_blocks = 1024;
    int32_t block_reduce_threads_per_block = 256;
    int32_t block_reduce_blocks =
            (block_reduce_num_blocks + block_reduce_threads_per_block - 1) / block_reduce_threads_per_block;
    double h_output[6];
    if (cuda_config.dim == 3) {
        WarpCompressorDouble::sampleMinMaxXYZKernel<<<num_blocks, sample_length, shared_mem_bytes>>>(
                d_x, d_y, d_z,
                d_min_x, d_min_y, d_min_z,
                d_max_x, d_max_y, d_max_z,
                sample_length, stride_between_samples, cuda_config.coordinate_num);

        WarpCompressorDouble::reduceMinMax<<<block_reduce_blocks, block_reduce_threads_per_block, 6 * block_reduce_threads_per_block *
                                                                            sizeof(double)>>>(
                d_min_x, d_max_x,
                d_min_y, d_max_y,
                d_min_z, d_max_z,
                d_output_minmax,
                num_blocks
        );
        cudaMemcpy(h_output, d_output_minmax, sizeof(double) * 6, cudaMemcpyDeviceToHost);
    } else {
        WarpCompressorDouble::sampleMinMaxXYZKernel2D<<<num_blocks, sample_length, shared_mem_bytes>>>(
                d_x, d_y,
                d_min_x, d_min_y, d_min_z,
                d_max_x, d_max_y, d_max_z,
                sample_length, stride_between_samples, cuda_config.coordinate_num);

        WarpCompressorDouble::reduceMinMax2D<<<block_reduce_blocks, block_reduce_threads_per_block, 6 * block_reduce_threads_per_block *
                                                                            sizeof(double)>>>(
                d_min_x, d_max_x,
                d_min_y, d_max_y,
                d_output_minmax,
                num_blocks
        );
        cudaMemcpy(h_output, d_output_minmax, sizeof(double) * 4, cudaMemcpyDeviceToHost);
        h_output[4] = 0;
        h_output[5] = 0;
    }

    double min_x = h_output[0];
    double max_x = h_output[1];
    double min_y = h_output[2];
    double max_y = h_output[3];
    double min_z = h_output[4];
    double max_z = h_output[5];

    double dx = max_x - min_x;
    double dy = max_y - min_y;
    double dz = max_z - min_z;
    double mean = (dx + dy + dz) / 3.0f;
    double variance = ((dx - mean) * (dx - mean) +
                      (dy - mean) * (dy - mean) +
                      (dz - mean) * (dz - mean)) / 3.0f;
    double std_dev = sqrt(variance);
    double cubic_ratio = std_dev / mean;
//    printf("Cubic ratio: %.6f\n", cubic_ratio);

    double max_range = 0;
    max_range = std::fmax(dx, dy);
    max_range = std::fmax(max_range, dz);
    if (cuda_config.seg_size == 0) {
        if (cubic_ratio < 0.1f) {
            if (cuda_config.error_mode == "rel") {
                if (cuda_config.rel_eb_dbl <= 1e-4) {
                    cuda_config.seg_size = 32;
                } else {
                    cuda_config.seg_size = 2;
                }
            }
        } else {
            if (cuda_config.error_mode == "rel") {
                if (cuda_config.rel_eb_dbl <= 1e-4) {
                    cuda_config.seg_size = 4;
                } else {
                    cuda_config.seg_size = 1;
                }
            }
        }
    }
    uint64_t max_blocks = static_cast<uint64_t>(ceil(max_range / (2 * cuda_config.seg_size * cuda_config.eb_dbl))) + 1;
    uint64_t total_block_id = max_blocks * max_blocks * max_blocks;
    bool adjusted_seg = total_block_id > UINT64_MAX;

    if (adjusted_seg == 1) {
        uint32_t temp_segment_size = cuda_config.seg_size;
        for (int32_t i = 0; i < 11; i++) {
            temp_segment_size *= 2;
            max_blocks = static_cast<uint64_t>(ceil(max_range / (2 * temp_segment_size *  cuda_config.eb_dbl))) + 1;
            total_block_id = max_blocks * max_blocks * max_blocks;
            if (total_block_id <= UINT64_MAX) {
                break;
            }
        }

        if (temp_segment_size > 1024) {
            printf("Error: The auto changed segment size %d is too large, please adjust your error bound.\n",
                   temp_segment_size);
            exit(1);
        }
        cuda_config.seg_size = temp_segment_size;
        cuda_config.adjusted_seg = true;
    }

    cudaFree(d_output_minmax);
}

void CudaWarpCompressDouble::compress(void *d_xyz, unsigned char *comp_data, uint64_t *comp_data_size,
                                      const std::string &comp_type) {
    auto* xyz_d = reinterpret_cast<double*>(d_xyz);
    compressImpl(xyz_d, comp_data, comp_data_size, comp_type);
}

void
CudaWarpCompressDouble::decompress(void *decData_x, void *decData_y, void *decData_z, unsigned char *d_compressed_data,
                                   const std::string &nvcomp_type) {
    auto* x = reinterpret_cast<double*>(decData_x);
    auto* y = reinterpret_cast<double*>(decData_y);
    auto* z = reinterpret_cast<double*>(decData_z);
    decompressImpl(x, y, z, d_compressed_data, nvcomp_type);
}

void
CudaWarpCompressDouble::compressImpl(double *d_xyz, unsigned char *d_comp_data, uint64_t *comp_size,
                           const std::string &comp_format) {
    // Mark the start of the compress function
    nvtxRangePushA("compress");

#ifdef GPZ_USE_NVCOMP
    const int32_t nvcomp_chunk_size = 1 << 16;
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;
    std::shared_ptr<nvcompManagerBase> nvcomp_manager;
    uint8_t *nvcomp_comp_buffer;

    if (comp_format == "lz4") {
        nvcomp_manager = std::make_shared<LZ4Manager>(nvcomp_chunk_size, nvcompBatchedLZ4Opts_t{data_type});
    } else if (comp_format == "snappy") {
        nvcomp_manager = std::make_shared<SnappyManager>(nvcomp_chunk_size, nvcompBatchedSnappyOpts_t{});
    } else if (comp_format == "bitcomp") {
        nvcomp_manager = std::make_shared<BitcompManager>(nvcomp_chunk_size,
                                                          nvcompBatchedBitcompFormatOpts{0 /* algo--fixed for now */,
                                                                                         data_type});
    } else if (comp_format == "ans") {
//        nvcompANSDataType_t ans_data_type = nvcompANSDataType_t::uint8;
        nvcomp_manager = std::make_shared<ANSManager>(nvcomp_chunk_size,
                                                      nvcompBatchedANSOpts_t{nvcomp_rANS});
    } else if (comp_format == "cascaded") {
        nvcompBatchedCascadedOpts_t cascaded_opts = nvcompBatchedCascadedDefaultOpts;
        cascaded_opts.type = data_type;
        cascaded_opts.chunk_size = nvcomp_chunk_size;
//        cascaded_opts.internal_chunk_bytes = nvcomp_chunk_size;
        nvcomp_manager = std::make_shared<CascadedManager>(nvcomp_chunk_size, cascaded_opts);
    } else if (comp_format == "gdeflate") {
        nvcomp_manager = std::make_shared<GdeflateManager>(nvcomp_chunk_size,
                                                           nvcompBatchedGdeflateOpts_t{0 /* algo--fixed for now */});
    } else if (comp_format == "deflate") {
        nvcomp_manager = std::make_shared<DeflateManager>(nvcomp_chunk_size, nvcompBatchedDeflateDefaultOpts);
    } else if (comp_format == "zstd") {
        // Get file size
        nvcomp_manager = std::make_shared<ZstdManager>(static_cast<uint64_t>(nvcomp_chunk_size),
                                                       nvcompBatchedZstdDefaultOpts);
    }
#else
    if (comp_format != "none" && !comp_format.empty()) {
        fprintf(stderr, "Error: nvcomp support is not compiled in. Rebuild with -DNVCOMP_DIR=<path> to enable nvcomp.\n");
        exit(1);
    }
#endif

    // alloc unique bid num for each block
    uint32_t grid_size = (cuda_config.coordinate_num + BLOCK_SIZE * CHUNK_SIZE - 1) /
                             (BLOCK_SIZE * CHUNK_SIZE);
    dim3 gridDim(grid_size);
    dim3 blockDim(BLOCK_SIZE);

    nvtxRangePushA("run compress kernel");

    uint64_t *d_cmp_offset;
    uint64_t *d_loc_offset;

    cudaMalloc(&d_cmp_offset, sizeof(uint64_t) * (grid_size));
    cudaMalloc(&d_loc_offset, sizeof(uint64_t) * (grid_size));
    cudaMemset(d_cmp_offset, 0, sizeof(uint64_t) * (grid_size));
    cudaMemset(d_loc_offset, 0, sizeof(uint64_t) * (grid_size));

    double *d_x = d_xyz;
    double *d_y = d_xyz + cuda_config.coordinate_num;
    double *d_z = d_xyz + 2 * cuda_config.coordinate_num;

    double *d_cache = reinterpret_cast<double *>(d_comp_data);
    // auto select seg size
    computeSegSize(d_x, d_y, d_z, d_cache, cuda_config);
    if (cuda_config.dim == 3) {
        // get equal length of compressed data
        WarpCompressorDouble::runCompressKernel<<<gridDim, blockDim>>>(d_x, d_y, d_z, d_comp_data, d_loc_offset,
                cuda_config.seg_size,
                cuda_config.coordinate_num,
                cuda_config.eb_dbl);
    } else {
        // get equal length of compressed data
        WarpCompressorDouble::runCompressKernel2D<<<gridDim, blockDim>>>(d_x, d_y, d_comp_data, d_loc_offset,
                cuda_config.seg_size,
                cuda_config.coordinate_num,
                cuda_config.eb_dbl);
    }


    void *d_temp_storage = nullptr;
    uint64_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_loc_offset, d_cmp_offset, grid_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_loc_offset, d_cmp_offset, grid_size);

    // use the input memory space as output
    unsigned char *d_xyz_out = reinterpret_cast<unsigned char *>(d_xyz);

    // copy prefix sum offset and compression parameters (coordinate number - 8 bytes, eb - 4 bytes, Seg Size 4 bytes)
    uint64_t prefix_sum_offset = grid_size * sizeof(uint64_t);
    uint32_t parameter_size = 32;
    cudaMemcpy(d_xyz_out, &cuda_config.dim, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyz_out + 4, &cuda_config.coordinate_num, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyz_out + 12, &cuda_config.float_bits, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyz_out + 16, &cuda_config.eb_dbl, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyz_out + 24, &cuda_config.seg_size, 4, cudaMemcpyHostToDevice);
    // padding 4 bytes for alignment
    cudaMemcpy(d_xyz_out + 28, &parameter_size, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyz_out + 32, d_cmp_offset, prefix_sum_offset, cudaMemcpyDeviceToDevice);
    // copy meta data
    const uint64_t meta_offset = grid_size * META_DATA_BYTES;
    cudaMemcpy(d_xyz_out + parameter_size + prefix_sum_offset, d_comp_data, meta_offset, cudaMemcpyDeviceToDevice);

    // compact compressed data
    WarpCompressorDouble::runCompactKernel<<<grid_size, 256>>>(d_comp_data + meta_offset, d_xyz_out + parameter_size +
                                                                                    prefix_sum_offset +
                                                                                    meta_offset, d_cmp_offset, d_loc_offset);

    // end run kernel
    nvtxRangePop();
    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    // get final compressed size
    // get compressed data size
    uint64_t last_cmp = 0, last_loc = 0;
    cudaMemcpy(&last_cmp, d_cmp_offset + (grid_size - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_loc, d_loc_offset + (grid_size - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t total = last_cmp + last_loc;

    *comp_size = total + parameter_size + prefix_sum_offset + meta_offset;

#ifdef GPZ_USE_NVCOMP
    uint64_t temp_comp_size = *comp_size;
    if (nvcomp_manager != nullptr) {
        // start malloc for nvcomp
        nvtxRangePushA("malloc for nvcomp");
        CompressionConfig comp_config = nvcomp_manager->configure_compression(temp_comp_size);
        CHECK_GPU(cudaMalloc(&nvcomp_comp_buffer, comp_config.max_compressed_buffer_size));

        // end malloc for nvcomp
        nvtxRangePop();

        nvtxRangePushA("run nvcomp compress kernel");
        nvcomp_manager->compress(d_xyz_out, nvcomp_comp_buffer, comp_config);
        nvtxRangePop();
        CHECK_GPU(cudaPeekAtLastError());
        CHECK_GPU(cudaDeviceSynchronize());
    }

    if (nvcomp_manager != nullptr) {
        uint64_t h_compressed_size = nvcomp_manager->get_compressed_output_size(nvcomp_comp_buffer);
        *comp_size = h_compressed_size;
        CHECK_GPU(cudaMemcpy(d_xyz_out, nvcomp_comp_buffer,
                             h_compressed_size * sizeof(unsigned char), cudaMemcpyDeviceToDevice));
    }
#endif
    // End of compress function
    nvtxRangePop();
    cudaFree(d_temp_storage);
    cudaFree(d_cmp_offset);
    cudaFree(d_loc_offset);

#ifdef GPZ_USE_NVCOMP
    if (nvcomp_manager != nullptr) {
        cudaFree(nvcomp_comp_buffer);
    }
#endif
}

void
CudaWarpCompressDouble::decompressImpl(double *decData_x, double *decData_y, double *decData_z, unsigned char *d_compressed_data,
                             const std::string &nvcomp_type) {
    // Mark the start of the compress function
    nvtxRangePushA("decompress");

    // nvcomp configuration
#ifdef GPZ_USE_NVCOMP
    if (!nvcomp_type.empty()) {
        const int32_t nvcomp_chunk_size = 1 << 16;
        nvcompType_t data_type = NVCOMP_TYPE_CHAR;
        std::shared_ptr<nvcompManagerBase> nvcomp_manager;

        if (nvcomp_type == "lz4") {
            nvcomp_manager = std::make_shared<LZ4Manager>(nvcomp_chunk_size, nvcompBatchedLZ4Opts_t{data_type});
        } else if (nvcomp_type == "snappy") {
            nvcomp_manager = std::make_shared<SnappyManager>(nvcomp_chunk_size, nvcompBatchedSnappyOpts_t{});
        } else if (nvcomp_type == "bitcomp") {
            nvcomp_manager = std::make_shared<BitcompManager>(nvcomp_chunk_size,
                                                              nvcompBatchedBitcompFormatOpts{
                                                                      0 /* algo--fixed for now */,
                                                                      data_type});
        } else if (nvcomp_type == "ans") {
//            nvcompANSDataType_t ans_data_type = nvcompANSDataType_t::uint8;
            nvcomp_manager = std::make_shared<ANSManager>(nvcomp_chunk_size,
                                                          nvcompBatchedANSOpts_t{nvcomp_rANS});
        } else if (nvcomp_type == "cascaded") {
            nvcompBatchedCascadedOpts_t cascaded_opts = nvcompBatchedCascadedDefaultOpts;
            cascaded_opts.type = data_type;
            cascaded_opts.chunk_size = nvcomp_chunk_size;
//            cascaded_opts.internal_chunk_bytes = nvcomp_chunk_size;
            nvcomp_manager = std::make_shared<CascadedManager>(nvcomp_chunk_size, cascaded_opts);
        } else if (nvcomp_type == "gdeflate") {
            nvcomp_manager = std::make_shared<GdeflateManager>(nvcomp_chunk_size,
                                                               nvcompBatchedGdeflateOpts_t{
                                                                       0 /* algo--fixed for now */});
        } else if (nvcomp_type == "deflate") {
            nvcomp_manager = std::make_shared<DeflateManager>(nvcomp_chunk_size, nvcompBatchedDeflateDefaultOpts);
        } else if (nvcomp_type == "zstd") {
            // Get file size
            nvcomp_manager = std::make_shared<ZstdManager>(static_cast<uint64_t>(nvcomp_chunk_size),
                                                            nvcompBatchedZstdDefaultOpts);
        }
        if (nvcomp_manager != nullptr) {
            DecompressionConfig decomp_config = nvcomp_manager->configure_decompression(d_compressed_data);
            uint8_t *res_decomp_buffer;
            CHECK_GPU(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));
            nvcomp_manager->decompress(res_decomp_buffer, d_compressed_data, decomp_config);
            d_compressed_data = res_decomp_buffer;
        }
    }
#else
    if (!nvcomp_type.empty() && nvcomp_type != "none") {
        fprintf(stderr, "Error: nvcomp support is not compiled in. Rebuild with -DNVCOMP_DIR=<path> to enable nvcomp.\n");
        exit(1);
    }
#endif

    // get compression parameters
    uint32_t parameter_size = 32;
    uint32_t dim = 0;
    uint64_t h_coordinate_num = 0;
    double h_eb = 0.0f;
    uint32_t h_seg_size = 0;
    cudaMemcpy(&dim, d_compressed_data, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_coordinate_num, d_compressed_data + 4, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_eb, d_compressed_data + 16, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_seg_size, d_compressed_data + 24, 4, cudaMemcpyDeviceToHost);

    // alloc unique bid num for each block
    uint32_t grid_size = (h_coordinate_num + BLOCK_SIZE * CHUNK_SIZE - 1) /
                             (BLOCK_SIZE * CHUNK_SIZE);


    nvtxRangePushA("run decompress kernel");
    dim3 gridDim(grid_size);
    dim3 blockDim(BLOCK_SIZE);

    uint64_t *d_cmp_offset;

    cudaMalloc(&d_cmp_offset, sizeof(uint64_t) * (grid_size));
    cudaMemset(d_cmp_offset, 0, sizeof(uint64_t) * (grid_size));

    // get prefix sum offset
    uint64_t prefix_sum_offset = grid_size * sizeof(uint64_t);
    cudaMemcpy(d_cmp_offset, d_compressed_data + parameter_size, prefix_sum_offset, cudaMemcpyDeviceToDevice);

    if (dim == 3) {
        // decompress
        WarpCompressorDouble::runDecompressKernel<<<gridDim, blockDim>>>(decData_x, decData_y, decData_z, d_compressed_data +
                                                                                                    parameter_size +
                                                                                                    prefix_sum_offset, d_cmp_offset,
                h_seg_size,
                h_coordinate_num,
                h_eb);
    } else {
        // decompress
        WarpCompressorDouble::runDecompressKernel2D<<<gridDim, blockDim>>>(decData_x, decData_y, d_compressed_data +
                                                                                                        parameter_size +
                                                                                                        prefix_sum_offset, d_cmp_offset,
                h_seg_size,
                h_coordinate_num,
                h_eb);
    }

    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());
    nvtxRangePop();

    // End of decompress function
    nvtxRangePop();

    cudaFree(d_cmp_offset);
}