#ifndef CUDA_CPMPRESS_H
#define CUDA_CPMPRESS_H

#include <string>
#include <cstdint>
// do not change
static const uint32_t FIND_M_CHUNK_SIZE = 64;
static const uint32_t FIND_M_BLOCK_SIZE = 32;
static const uint32_t CHUNK_SIZE = 32;
static const uint32_t BLOCK_SIZE = 32;
const int ELEMS_PER_BLOCK = BLOCK_SIZE * CHUNK_SIZE;

// using for lc, threads per block [must be power of 2 and at least 128]
static const int TPB = 512;

class CudaConfig {
public:
    uint32_t seg_size;
    bool adjusted_seg;
    uint32_t dim;
    uint64_t coordinate_num;
    float eb;
    float rel_eb;
    double eb_dbl;
    double rel_eb_dbl;
    std::string error_mode;
    uint32_t float_bits;

    CudaConfig() {}

    CudaConfig(uint32_t seg_size, bool adjusted_seg, uint32_t dim, uint64_t coordinate_num, std::string error_mode, float eb, float rel_eb, uint32_t float_bits) :
            seg_size(seg_size), adjusted_seg(adjusted_seg), dim(dim), coordinate_num(coordinate_num),
            error_mode(std::move(error_mode)), eb(eb), rel_eb(rel_eb), float_bits(float_bits) {
    }

    CudaConfig(uint32_t seg_size, bool adjusted_seg, uint32_t dim, uint64_t coordinate_num, std::string error_mode, double eb, double rel_eb, uint32_t float_bits) :
            seg_size(seg_size), adjusted_seg(adjusted_seg), dim(dim), coordinate_num(coordinate_num),
            error_mode(std::move(error_mode)), eb_dbl(eb), rel_eb_dbl(rel_eb), float_bits(float_bits) {
    }
};

#endif
