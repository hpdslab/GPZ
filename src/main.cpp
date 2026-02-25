#include <cmath>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.cuh"
#include <cstring>
#include <thread>
#include <getopt.h>
#include "CompressorManager.h"
#include "cudaWarpCompress.cuh"
#include "cudaTimer.cuh"
#include "CpuTimer.hpp"
#include "cudaWarpCompressDouble.cuh"

void print_decompression_result(uint64_t decompressed_num_points, double decompression_duration_ms, uint32_t dim) {
    printf("Decompression_CUDA throughput (GB/s): %f\n",
           (decompressed_num_points * dim * sizeof(float) / 1024 / 1024 / 1024.0) /
           (decompression_duration_ms / 1000.0));
    printf("Decompression_CUDA execution time (ms): %f\n", decompression_duration_ms);
}

void print_compression_result(uint64_t comp_size, uint64_t orignal_file_bytes, double g_bytes,
                              float gpu_comprsssion_duration_milliseconds,
                              CudaConfig cuda_config) {
    printf("Segment size: %d\n", cuda_config.seg_size);
    printf("Adjusted Seg: %s\n", cuda_config.adjusted_seg ? "true" : "false");
    printf("PCompression size (MB): %f\n", comp_size / 1024.0 / 1024.0);
    printf("PCompression ratio: %f\n", (orignal_file_bytes * 1.0 / comp_size));
    printf("PCompression_CUDA throughput (GB/s): %f\n",
           (orignal_file_bytes / g_bytes) /
           (gpu_comprsssion_duration_milliseconds / 1000.0));
    printf("PCompression_CUDA execution time (ms): %f\n", gpu_comprsssion_duration_milliseconds);
}

template<typename T>
T convertRelEb(T error_bound, T* min_max_values) {
    T max_range = 0;

    printf("min_max_values: %.10g %.10g %.10g %.10g %.10g %.10g\n",
           min_max_values[0], min_max_values[1], min_max_values[2],
           min_max_values[3], min_max_values[4], min_max_values[5]);

    T x_range = min_max_values[3] - min_max_values[0];
    T y_range = min_max_values[4] - min_max_values[1];
    T z_range = min_max_values[5] - min_max_values[2];

    max_range = std::fmax(x_range, y_range);
    max_range = std::fmax(max_range, z_range);

    error_bound = max_range * error_bound;

    printf("Max Range is: %.10g\n", max_range);
    printf("Abs Error bound: %.10g\n", error_bound);
    return error_bound;
}



void print_usage(const char *prog_name) {
    std::cerr << "\nUsage:\n"
              << "  " << prog_name
              << " -x -i <input_file> -o <output_file> -n <num_points> -e <error_bound> -s <segment_size> -m <error_mode> [-t <nvcomp_type>]\n"
              << "      Compress mode:\n"
              << "        -x               Enable compression mode\n"
              << "        -i <file>        Input file to compress (required)\n"
              << "        -o <file>        Output compressed file (required)\n"
              << "        -p <dim>         [Optional]Number of dimensions (only support 2 or 3) (default: 3)\n"
              << "        -n <num>         Number of points (required)\n"
              << "        -m <string>      Error mode (e.g., rel, abs) (required)\n"
              << "        -f <int>         Float bits (32 or 64)\n"
              << "        -e <float>       Error bound (required)\n"
              << "        -s <int>         [Optional] Segment size\n"
              << "        -t <string>      [Optional] nvcomp compression type (e.g., none, lz4)\n"
              << "\n"
              << "  " << prog_name
              << " -d -i <input_file> -o <output_file> [-r <recontracted_file>] [-v <original_file>] [-t <nvcomp_type> -n <num_points> -e <error_bound> -f <float_bits>]\n"
              << "      Decompress mode:\n"
              << "        -d               Enable decompression mode\n"
              << "        -o <file>        Output decompressed file (required)\n"
              << "        -r <file>        Output recontracted (refitted) file\n"
              << "        -v <file>        [Optional] Verification mode, compares output to original file\n"
              << "        -t <string>      [Optional] nvcomp type; if set, requires -n, -e, and -f\n"
              << "        -n <num>         [Required if -t used] Number of points\n"
              << "        -e <float>       [Required if -t used] Error bound\n"
              << "        -f <int>         [Required if -t used] Float bits (32 or 64)\n"
              << "        -p <dim>         [Optional] Number of dimensions (default: 3)\n"
              << std::endl;
}

int32_t main(int32_t argc, char *argv[]) {
    const char *filename = nullptr;
    const char *output_filename = nullptr;
    const char *recontracted_filename = nullptr;
    const char *original_filename = nullptr;

    uint64_t numPoints = 0;
    uint32_t float_bits = 0;
    uint32_t dim = 3;
    double error_bound = 0.0f;
    uint32_t segment_size = 0;
    std::string err_mode;
    std::string nvcomp_type = "none";

    bool compress_mode = false;
    bool decompress_mode = false;
    bool verification_mode = false;


    int32_t opt;
    while ((opt = getopt(argc, argv, "xdi:o:p:n:e:s:m:f:t:r:v:h")) != -1) {
        switch (opt) {
            case 'x':
                compress_mode = true;
                break;
            case 'd':
                decompress_mode = true;
                break;
            case 'i':
                filename = optarg;
                break;
            case 'o':
                output_filename = optarg;
                break;
            case 'p':
                dim = std::stoul(optarg);
                break;
            case 'n':
                numPoints = std::stoull(optarg);
                break;
            case 'e':
                error_bound = std::stod(optarg);
                break;
            case 's':
                segment_size = std::stoul(optarg);
                break;
            case 'm':
                err_mode = optarg;
                break;
            case 'f':
                float_bits = std::stoul(optarg);
                break;
            case 't':
                nvcomp_type = optarg;
                break;
            case 'r':
                recontracted_filename = optarg;
                break;
            case 'v':
                verification_mode = true;
                original_filename = optarg;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    if (compress_mode && decompress_mode) {
        std::cerr << "Error: Cannot specify both -x (compress) and -d (decompress)\n";
        return 1;
    }

    if (!compress_mode && !decompress_mode) {
        std::cerr << "Error: Must specify either -x (compress) or -d (decompress)\n";
        print_usage(argv[0]);
        return 1;
    }

    if (compress_mode) {
        if (!filename || !output_filename || numPoints == 0 || err_mode.empty()) {
            std::cerr << "Error: Missing required arguments for compression\n";
            print_usage(argv[0]);
            return 1;
        }
        if (dim != 2 && dim != 3) {
            std::cerr << "Error: Dimension must be 2 or 3\n";
            print_usage(argv[0]);
            return 1;
        }
        if (float_bits != 32 && float_bits != 64) {
            std::cerr << "Error: Float bits must be 32 or 64\n";
            print_usage(argv[0]);
            return 1;
        }
#ifndef GPZ_USE_NVCOMP
        if (nvcomp_type != "none" && !nvcomp_type.empty()) {
            std::cerr << "Error: -t option requires nvcomp support. Rebuild with -DNVCOMP_DIR=<path> to enable.\n";
            return 1;
        }
#endif
    }

    if (decompress_mode) {
        if (!recontracted_filename || !output_filename) {
            std::cerr << "Error: -i and -o are required for decompression\n";
            print_usage(argv[0]);
            return 1;
        }

        if (verification_mode) {
            if (!original_filename) {
                std::cerr << "Error: -v requires an original file name\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        if (nvcomp_type != "none" && !nvcomp_type.empty()) {
#ifdef GPZ_USE_NVCOMP
            if (numPoints == 0 || error_bound == 0.0f || (float_bits != 32 && float_bits != 64)) {
                std::cerr << "Error: When -t (using nvcomp) is provided, -n, -e, and -f are also required\n";
                print_usage(argv[0]);
                return 1;
            }
#else
            std::cerr << "Error: -t option requires nvcomp support. Rebuild with -DNVCOMP_DIR=<path> to enable.\n";
            return 1;
#endif
        }
    }

    double input_error_bound = error_bound;
#ifdef GPZ_USE_NVCOMP
    printf("NVCOMP Algorithm: %s\n", nvcomp_type.c_str());
#endif

    if (compress_mode) {
        void *h_x = nullptr, *h_y = nullptr, *h_z = nullptr;
        if (float_bits == 32) {
            load_file_to_soa_xyz<float>(filename,
                                        reinterpret_cast<float**>(&h_x),
                                        reinterpret_cast<float**>(&h_y),
                                        reinterpret_cast<float**>(&h_z),
                                        numPoints, dim);
        } else {
            load_file_to_soa_xyz<double>(filename,
                                         reinterpret_cast<double**>(&h_x),
                                         reinterpret_cast<double**>(&h_y),
                                         reinterpret_cast<double**>(&h_z),
                                         numPoints, dim);
        }

        if (h_x && h_y && h_z) {
            printf("Successfully read %ld points, %.03f MB from %s\n", numPoints, numPoints * dim * (
                    float_bits == 32 ? 4 : 8) / 1024.0 / 1024.0, filename);
            if (err_mode == "rel") {
                void* min_max_values = nullptr;
                if (float_bits == 32) {
                    min_max_values = static_cast<void*>(findMinimumValues<float>(
                            reinterpret_cast<float*>(h_x),
                            reinterpret_cast<float*>(h_y),
                            reinterpret_cast<float*>(h_z),
                            numPoints));
                    error_bound = convertRelEb<float>((float)error_bound, reinterpret_cast<float*>(min_max_values));

                    delete[] static_cast<float*>(min_max_values);
                } else {
                    min_max_values = static_cast<void*>(findMinimumValues<double>(
                            reinterpret_cast<double*>(h_x),
                            reinterpret_cast<double*>(h_y),
                            reinterpret_cast<double*>(h_z),
                            numPoints));
                    error_bound = convertRelEb<double>(error_bound, reinterpret_cast<double*>(min_max_values));
                    delete[] static_cast<double*>(min_max_values);
                }
            }

            /* Compress module */
            ICudaCompressor* compressor;
            if (float_bits == 32) {
                CudaConfig cuda_config(segment_size, false, dim, numPoints, err_mode, (float)error_bound, (float)input_error_bound, float_bits);
                compressor = new CudaWarpCompress(cuda_config);
            } else {
                CudaConfig cuda_config_double(segment_size, false, dim, numPoints, err_mode, error_bound, input_error_bound, float_bits);
                compressor = new CudaWarpCompressDouble(cuda_config_double);
            }

            // compressed size
            uint64_t comp_size = 0;

            // allocate host memory for storing result
            unsigned char *h_comp_data = nullptr;
            uint64_t d_xyz_size = float_bits == 32 ? numPoints * 3 * sizeof(float) :
                                  numPoints * 3 * sizeof(double);
            cudaMallocHost((void **) &h_comp_data, d_xyz_size);

            // allocate device memory for storing temporary data
            unsigned char *d_comp_buff;
            uint64_t d_comp_buff_size = float_bits == 32 ? numPoints * 4 * sizeof(float) :
                                        numPoints * 4 * sizeof(double);
            CHECK_GPU(cudaMalloc((void **) &d_comp_buff, d_comp_buff_size));

            // allocate device memory for input and final output (compressed data), init compressor
            void *d_xyz;
            cudaMalloc(&d_xyz, d_xyz_size);
            auto *compressorManager = new CompressorManager();



            // warm up
            for (int32_t i = 0; i < 5; i++) {
                uint64_t dummy_size = 0;
                if (float_bits == 32) {
                    cudaMemcpy((float*)d_xyz, h_x, sizeof(float) * numPoints, cudaMemcpyHostToDevice);
                    cudaMemcpy((float*)d_xyz + numPoints, h_y, sizeof(float) * numPoints,
                               cudaMemcpyHostToDevice);
                    cudaMemcpy((float*)d_xyz + 2 * numPoints, h_z, sizeof(float) * numPoints,
                               cudaMemcpyHostToDevice);
                    compressorManager->compressWarp((float*)d_xyz, d_comp_buff, &dummy_size, nvcomp_type, compressor);
                } else {
                    cudaMemcpy((double*)d_xyz, h_x, sizeof(double) * numPoints, cudaMemcpyHostToDevice);
                    cudaMemcpy((double*)d_xyz + numPoints, h_y, sizeof(double) * numPoints,
                               cudaMemcpyHostToDevice);
                    cudaMemcpy((double*)d_xyz + 2 * numPoints, h_z, sizeof(double) * numPoints,
                               cudaMemcpyHostToDevice);
                    compressorManager->compressWarp((double*)d_xyz, d_comp_buff, &dummy_size, nvcomp_type, compressor);
                }
                cudaMemset(d_comp_buff, 0, d_comp_buff_size);
                cudaMemset(d_xyz, 0, d_xyz_size);
            }

            if (float_bits == 32) {
                cudaMemcpy((float*)d_xyz, h_x, sizeof(float) * numPoints, cudaMemcpyHostToDevice);
                cudaMemcpy((float*)d_xyz + numPoints, h_y, sizeof(float) * numPoints,
                           cudaMemcpyHostToDevice);
                cudaMemcpy((float*)d_xyz + 2 * numPoints, h_z, sizeof(float) * numPoints,
                           cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy((double*)d_xyz, h_x, sizeof(double) * numPoints, cudaMemcpyHostToDevice);
                cudaMemcpy((double*)d_xyz + numPoints, h_y, sizeof(double) * numPoints,
                           cudaMemcpyHostToDevice);
                cudaMemcpy((double*)d_xyz + 2 * numPoints, h_z, sizeof(double) * numPoints,
                           cudaMemcpyHostToDevice);
            }

            CudaTimer gpu_timer;
            gpu_timer.start();

            // start compress
            if (float_bits == 32) {
                compressorManager->compressWarp((float*)d_xyz, d_comp_buff, &comp_size, nvcomp_type, compressor);
            } else {
                compressorManager->compressWarp((double*)d_xyz, d_comp_buff, &comp_size, nvcomp_type, compressor);
            }

            gpu_timer.stop();

            CHECK_GPU(cudaMemcpy(h_comp_data, d_xyz, comp_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

            // write to file
            write_unsigned_char_to_file(h_comp_data, comp_size, output_filename);

            /* Compress end */
            uint64_t orignal_file_bytes = float_bits == 32 ? numPoints * dim * sizeof(float) :
                                          numPoints * dim * sizeof(double);
            double gbytes = 1024.0 * 1024.0 * 1024.0;
            auto gpu_comprsssion_duration_milliseconds = gpu_timer.elapsedMilliseconds();

            print_compression_result(comp_size, orignal_file_bytes, gbytes,
                                     gpu_comprsssion_duration_milliseconds,
                                     compressor->getConfig());

            // free cuda host
            cudaFreeHost(h_x);
            cudaFreeHost(h_y);
            cudaFreeHost(h_z);
            cudaFreeHost(h_comp_data);

            cudaFree(d_xyz);
            cudaFree(d_comp_buff);
            delete compressor;
        }
        exit(0);
    }

    if (decompress_mode) {
        unsigned char *h_compressed_data;
        uint64_t data_size;
        load_file_to_cuda_hostmem(output_filename, &h_compressed_data, &data_size);
        printf("Compressed file loaded, data size %f MB\n", data_size / 1024.0 / 1024.0);

        // init compressor and compress
        // load parameters
        uint32_t cli_dim = dim;  // save CLI-provided dim before reset (needed for nvcomp)
        dim = 0;
        uint64_t decompressed_num_points = 0;
        uint32_t decompressed_float_bits = 0;
        auto *decompressor = new CudaWarpCompress();
        auto *decompressorDouble = new CudaWarpCompressDouble();

        CompressorManager *compressorManager = new CompressorManager();

        CpuTimer preload_paras_timer;
        preload_paras_timer.start();
        compressorManager->decompressLoadParas(&dim, &decompressed_num_points, &decompressed_float_bits, nvcomp_type,
                                               numPoints, cli_dim, float_bits, h_compressed_data);
        printf("Decompressed num points: %zu\n", decompressed_num_points);
        preload_paras_timer.stop();

        // allocate host memory for storing result
        uint64_t single_dim_size = decompressed_float_bits == 32 ? sizeof(float) * decompressed_num_points :
                sizeof(double) * decompressed_num_points;
        void *h_dec_data_x = nullptr;
        void *h_dec_data_y = nullptr;
        void *h_dec_data_z = nullptr;
        cudaMallocHost(&h_dec_data_x, single_dim_size);
        cudaMallocHost(&h_dec_data_y, single_dim_size);
        cudaMallocHost(&h_dec_data_z, single_dim_size);

        // allocate device memory
        unsigned char *d_compressed_data;
        // if compression ratio is less than 3, d_compressed_data need to be larger than single_dim_size
        CHECK_GPU(cudaMalloc(&d_compressed_data, single_dim_size * 2));
        void *d_dec_data_x, *d_dec_data_y, *d_dec_data_z;
        cudaMalloc(&d_dec_data_x, single_dim_size);
        cudaMalloc(&d_dec_data_y, single_dim_size);
        cudaMalloc(&d_dec_data_z, single_dim_size);

        // warm up
        for (int32_t i = 0; i < 5; i++) {
            CHECK_GPU(cudaMemcpy(d_compressed_data, h_compressed_data, data_size * sizeof(unsigned char),
                                 cudaMemcpyHostToDevice));
            if (decompressed_float_bits == 32) {
                compressorManager->decompressWarp((float*)d_dec_data_x, (float*)d_dec_data_y, (float*)d_dec_data_z,
                                                  d_compressed_data, decompressor, nvcomp_type);
            } else {
                compressorManager->decompressWarp((double*)d_dec_data_x, (double*)d_dec_data_y, (double*)d_dec_data_z,
                                                  d_compressed_data, decompressorDouble, nvcomp_type);
            }
            cudaMemset(d_dec_data_x, 0, single_dim_size);
            cudaMemset(d_dec_data_y, 0, single_dim_size);
            cudaMemset(d_dec_data_z, 0, single_dim_size);
            cudaMemset(d_compressed_data, 0, single_dim_size * 2);
        }
        CHECK_GPU(cudaMemcpy(d_compressed_data, h_compressed_data, data_size * sizeof(unsigned char),
                             cudaMemcpyHostToDevice));

        CudaTimer timer_decompress;
        timer_decompress.start();

        if (decompressed_float_bits == 32) {
            compressorManager->decompressWarp((float*)d_dec_data_x, (float*)d_dec_data_y, (float*)d_dec_data_z,
                                              d_compressed_data, decompressor, nvcomp_type);
        } else {
            compressorManager->decompressWarp((double*)d_dec_data_x, (double*)d_dec_data_y, (double*)d_dec_data_z,
                                              d_compressed_data, decompressorDouble, nvcomp_type);
        }
        timer_decompress.stop();

        // copy decompressed data to host
        CHECK_GPU(cudaMemcpy(h_dec_data_x, d_dec_data_x, single_dim_size, cudaMemcpyDeviceToHost));
        CHECK_GPU(cudaMemcpy(h_dec_data_y, d_dec_data_y, single_dim_size, cudaMemcpyDeviceToHost));
        CHECK_GPU(cudaMemcpy(h_dec_data_z, d_dec_data_z, single_dim_size, cudaMemcpyDeviceToHost));

        /* Decompress end */

        // write to file
        if (decompressed_float_bits == 32) {
            write_xyz_to_aos_file<float>(reinterpret_cast<float*>(h_dec_data_x),
                    reinterpret_cast<float*>(h_dec_data_y),
                    reinterpret_cast<float*>(h_dec_data_z),
                    decompressed_num_points, recontracted_filename, dim);
        } else {
            write_xyz_to_aos_file<double>(reinterpret_cast<double*>(h_dec_data_x),
                    reinterpret_cast<double*>(h_dec_data_y),
                    reinterpret_cast<double*>(h_dec_data_z),
                    decompressed_num_points, recontracted_filename, dim);
        }

        auto decompression_duration_ms =
                preload_paras_timer.elapsed_milliseconds() + timer_decompress.elapsedMilliseconds();

        print_decompression_result(decompressed_num_points, decompression_duration_ms, dim);

        /* verify by gpu and compute psnr */
        if (original_filename != nullptr && *original_filename != '\0') {
            void *h_x, *h_y, *h_z;
            void* min_max_values = nullptr;
            if (decompressed_float_bits == 32) {
                load_file_to_soa_xyz<float>(original_filename,
                                            reinterpret_cast<float**>(&h_x),
                                            reinterpret_cast<float**>(&h_y),
                                            reinterpret_cast<float**>(&h_z),
                                            decompressed_num_points, dim);
                min_max_values = static_cast<void*>(findMinimumValues<float>(
                        reinterpret_cast<float*>(h_x),
                        reinterpret_cast<float*>(h_y),
                        reinterpret_cast<float*>(h_z),
                        decompressed_num_points));
            } else {
                load_file_to_soa_xyz<double>(original_filename,
                                             reinterpret_cast<double**>(&h_x),
                                             reinterpret_cast<double**>(&h_y),
                                             reinterpret_cast<double**>(&h_z),
                                             decompressed_num_points, dim);
                min_max_values = static_cast<void*>(findMinimumValues<double>(
                        reinterpret_cast<double*>(h_x),
                        reinterpret_cast<double*>(h_y),
                        reinterpret_cast<double*>(h_z),
                        decompressed_num_points));
            }
            double verification_error_bound = 0.0f;
            uint32_t verification_dim = dim;
#ifdef GPZ_USE_NVCOMP
            if (!nvcomp_type.empty() && nvcomp_type != "none") {
                verification_error_bound = error_bound;
            } else {
#endif
                memcpy(&verification_dim, h_compressed_data, sizeof(uint32_t));
                if (decompressed_float_bits == 32) {
                    float temp_verification_error_bound = 0.0f;
                    memcpy(&temp_verification_error_bound, h_compressed_data + 16, sizeof(float));
                    verification_error_bound = temp_verification_error_bound;
                } else {
                    memcpy(&verification_error_bound, h_compressed_data + 16, sizeof(double));
                }
#ifdef GPZ_USE_NVCOMP
            }
#endif
            printf("Verification error bound: %.10g\n", verification_error_bound);

            if (decompressed_float_bits == 32) {
                verifyCUDA((float*)h_dec_data_x, (float*)h_dec_data_y, (float*)h_dec_data_z, (float*)h_x, (float*)h_y, (float*)h_z,
                                  decompressed_num_points, (float*)min_max_values, (float)verification_error_bound, dim);
                delete[] static_cast<float*>(min_max_values);
            } else {
                verifyCUDA((double*)h_dec_data_x, (double*)h_dec_data_y, (double*)h_dec_data_z, (double*)h_x, (double*)h_y, (double*)h_z,
                                  decompressed_num_points, (double*)min_max_values, verification_error_bound, dim);
                delete[] static_cast<double*>(min_max_values);
            }

            cudaFreeHost(h_x);
            cudaFreeHost(h_y);
            cudaFreeHost(h_z);
        }

        cudaFreeHost(h_dec_data_x);
        cudaFreeHost(h_dec_data_y);
        cudaFreeHost(h_dec_data_z);

        cudaFree(d_compressed_data);
        cudaFree(d_dec_data_x);
        cudaFree(d_dec_data_y);
        cudaFree(d_dec_data_z);

        delete decompressor;
        cudaFreeHost(h_compressed_data);
        delete compressorManager;
        delete decompressorDouble;

    }

    return 0;
}







