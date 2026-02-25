#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <limits>
#include <algorithm>
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#define CHECK_GPU(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static void CheckCuda(const int line) {
    cudaError_t e;
    cudaDeviceSynchronize();
    if (cudaSuccess != (e = cudaGetLastError())) {
        fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
        throw std::runtime_error("Debug error");
    }
}

__device__ __forceinline__ void check_alignment_device(const void *ptr, uint64_t alignment, const char *label) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr % alignment != 0) {
        printf("[DEVICE ALIGNMENT ERROR] %s is not %zu-byte aligned! Addr = %p (mod %lu = %lu)\n",
               label, alignment, ptr, alignment, addr % alignment);
        asm("trap;");
    }
}


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
T *findMinimumValues(const T *h_x, const T *h_y, const T *h_z, uint64_t size) {
    T *min_max_values = new T[6];

    min_max_values[0] = std::numeric_limits<float>::max();
    min_max_values[1] = std::numeric_limits<float>::max();
    min_max_values[2] = std::numeric_limits<float>::max();
    min_max_values[3] = std::numeric_limits<float>::lowest();
    min_max_values[4] = std::numeric_limits<float>::lowest();
    min_max_values[5] = std::numeric_limits<float>::lowest();

    for (uint64_t i = 0; i < size; i++) {
        min_max_values[0] = std::min(min_max_values[0], h_x[i]);
        min_max_values[1] = std::min(min_max_values[1], h_y[i]);
        min_max_values[2] = std::min(min_max_values[2], h_z[i]);
        min_max_values[3] = std::max(min_max_values[3], h_x[i]);
        min_max_values[4] = std::max(min_max_values[4], h_y[i]);
        min_max_values[5] = std::max(min_max_values[5], h_z[i]);
    }

    return min_max_values;
}

float *readSingleFrame(const std::string &filename, uint64_t numPoints, uint64_t timeFrameIndex);

void warm_up();

void printGPUMemoUsage(int num);

float *loadF32File(const std::string &filename, uint64_t &numFloats);

void write_unsigned_char_to_file(const unsigned char *h_encoded_bids, uint64_t length, const char *filename);

void write_uint64_t_to_file(const uint64_t *h_encoded_bids, uint64_t length, const char *filename);

void write_int_to_file(const int *h_encoded_bids, uint64_t length, const char *filename);

void load_file_to_cuda_hostmem(const char *filename, unsigned char **host_ptr, uint64_t *file_size);

template <typename T>
void load_file_to_soa_xyz(const char *filename, T **h_x, T **h_y, T **h_z, uint64_t num_points, uint32_t dim) {
    FILE *file = fopen(filename, "rb");
    if (!file) throw std::runtime_error("Failed to open input file");

    // Check file size
    fseek(file, 0, SEEK_END);
    uint64_t file_size = ftell(file);
    rewind(file);

    uint64_t point_size = dim * sizeof(T);
    if (file_size % point_size != 0) {
        fclose(file);
        throw std::runtime_error("File size is not aligned to expected T struct");
    }

    uint64_t actual_points = file_size / point_size;
    if (actual_points != num_points) {
        fclose(file);
        throw std::runtime_error("File size does not match expected number of points");
    }

    // Allocate pinned memory
    auto check = [](cudaError_t err) {
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    };
    check(cudaMallocHost(h_x, num_points * sizeof(T)));
    check(cudaMallocHost(h_y, num_points * sizeof(T)));
    check(cudaMallocHost(h_z, num_points * sizeof(T)));

    if (dim == 3) {
        struct XYZ { T x, y, z; };
        XYZ *temp = new (std::nothrow) XYZ[num_points];
        if (!temp) throw std::runtime_error("Memory allocation failed");
        if (fread(temp, sizeof(XYZ), num_points, file) != num_points) {
            delete[] temp; fclose(file);
            throw std::runtime_error("Failed to read complete 3D data");
        }
        for (uint64_t i = 0; i < num_points; ++i) {
            (*h_x)[i] = temp[i].x;
            (*h_y)[i] = temp[i].y;
            (*h_z)[i] = temp[i].z;
        }
        delete[] temp;
    } else if (dim == 2) {
        struct XY {
            T x, y;
        };
        XY *temp = new(std::nothrow) XY[num_points];
        if (!temp) throw std::runtime_error("Memory allocation failed");
        if (fread(temp, sizeof(XY), num_points, file) != num_points) {
            delete[] temp;
            fclose(file);
            throw std::runtime_error("Failed to read complete 2D data");
        }
        for (uint64_t i = 0; i < num_points; ++i) {
            (*h_x)[i] = temp[i].x;
            (*h_y)[i] = temp[i].y;
            (*h_z)[i] = static_cast<T>(0);
        }
        delete[] temp;
    }else {
        fclose(file);
        throw std::runtime_error("Unsupported dimension (only 2 or 3 allowed)");
    }

    fclose(file);
}

template<typename T>
void write_xyz_to_aos_file(const T* h_dec_data_x, const T* h_dec_data_y, const T* h_dec_data_z,
                           uint64_t N, const std::string& filename, uint32_t dim) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Failed to open file for writing");
    }

    if (dim == 3) {
        std::vector<T> interleaved(3 * N);
        for (uint64_t i = 0; i < N; ++i) {
            interleaved[3 * i + 0] = h_dec_data_x[i];
            interleaved[3 * i + 1] = h_dec_data_y[i];
            interleaved[3 * i + 2] = h_dec_data_z[i];
        }
        fout.write(reinterpret_cast<const char*>(interleaved.data()), interleaved.size() * sizeof(T));
    }

    if (dim == 2) {
        std::vector<T> interleaved(2 * N);
        for (uint64_t i = 0; i < N; ++i) {
            interleaved[2 * i + 0] = h_dec_data_x[i];
            interleaved[2 * i + 1] = h_dec_data_y[i];
        }
        fout.write(reinterpret_cast<const char*>(interleaved.data()), interleaved.size() * sizeof(T));
    }

    fout.close();
}

//void write_xyz_to_aos_file(const float *h_dec_data_x, const float *h_dec_data_y, const float *h_dec_data_z, uint64_t N,
//                           const std::string &filename, uint32_t dim);

void
verify(float *h_dec_data_x, float *h_dec_data_y, float *h_dec_data_z, float *h_x, float *h_y, float *h_z,
       uint64_t numPoints, float *min_max_values, uint64_t coordinate_num, float eb);

void verifyCUDA(float *h_dec_data_x, float *h_dec_data_y, float *h_dec_data_z, float *h_x, float *h_y, float *h_z,
                uint64_t coordinate_num, float *min_max_values, float eb, uint32_t dim);

void verifyCUDA(double *h_dec_data_x, double *h_dec_data_y, double *h_dec_data_z, double *h_x, double *h_y, double *h_z,
                uint64_t coordinate_num, double *min_max_values, double eb, uint32_t dim);
//double combine_psnr(const std::vector<double> &psnr_list);
//
//template <typename T>
//void verifyCUDA(T* h_dec_data_x, T* h_dec_data_y, T* h_dec_data_z,
//                T* h_x, T* h_y, T* h_z,
//                uint64_t coordinate_num, T* min_max_values, T eb, uint32_t dim) {
//    printf("verify by CUDA...\n");
//
//    auto [err_count_x, psnr_x] = verify_and_psnr(h_dec_data_x, h_x, coordinate_num, eb,
//                                                 min_max_values[0], min_max_values[3], "X");
//    auto [err_count_y, psnr_y] = verify_and_psnr(h_dec_data_y, h_y, coordinate_num, eb,
//                                                 min_max_values[1], min_max_values[4], "Y");
//
//    std::vector<double> psnrs = {psnr_x, psnr_y};
//    uint64_t total_error_count = err_count_x + err_count_y;
//
//    if (dim == 3) {
//        auto [err_count_z, psnr_z] = verify_and_psnr(h_dec_data_z, h_z, coordinate_num, eb,
//                                                     min_max_values[2], min_max_values[5], "Z");
//        psnrs.push_back(psnr_z);
//        total_error_count += err_count_z;
//    }
//
//    double psnr = combine_psnr(psnrs);
//
//    printf("Error Count: %zu\n", total_error_count);
//    printf("PSNR: %f\n", psnr);
//}

#endif