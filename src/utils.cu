#include "utils.cuh"

__global__ void warm_up_gpu() {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void warm_up() {
    warm_up_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(0);
}


__global__ void compute_error_and_mse(
        const float *dec, const float *orig,
        uint64_t N, float eb,
        uint32_t *d_error_count, double *d_mse_sum) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float diff = dec[idx] - orig[idx];
    float abs_diff = fabsf(diff);
    float squared = diff * diff;

    if (abs_diff > 1.01f * eb) {
        atomicAdd(d_error_count, 1);
    }

    atomicAdd(d_mse_sum, (double) squared);
}

__global__ void compute_error_and_mse_double(
        const double *dec, const double *orig,
        uint64_t N, double eb,
        uint32_t *d_error_count, double *d_mse_sum) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double diff = dec[idx] - orig[idx];
    double abs_diff = fabs(diff);
    double squared = diff * diff;

    if (abs_diff > 1.01f * eb) {
        atomicAdd(d_error_count, 1);
    }

    atomicAdd(d_mse_sum, (double) squared);
}

std::pair<uint32_t, float> verify_and_psnr(
        const float *h_dec_data, const float *h_ref_data,
        uint64_t coordinate_num, float eb,
        float min_val, float max_val,
        const char *label) {
    thrust::device_vector<float> d_dec(h_dec_data, h_dec_data + coordinate_num);
    thrust::device_vector<float> d_ref(h_ref_data, h_ref_data + coordinate_num);
    CheckCuda(__LINE__);
    thrust::sort(d_dec.begin(), d_dec.end());
    thrust::sort(d_ref.begin(), d_ref.end());
    CheckCuda(__LINE__);
    uint32_t *d_error_count;
    double *d_mse_sum;
    cudaMalloc(&d_error_count, sizeof(uint32_t));
    cudaMalloc(&d_mse_sum, sizeof(double));
    cudaMemset(d_error_count, 0, sizeof(uint32_t));
    cudaMemset(d_mse_sum, 0, sizeof(double));
    CheckCuda(__LINE__);
    int32_t threads = 256;
    uint64_t blocks = (coordinate_num + threads - 1) / threads;
    compute_error_and_mse<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_dec.data()),
            thrust::raw_pointer_cast(d_ref.data()),
            coordinate_num, eb,
            d_error_count, d_mse_sum);
    CheckCuda(__LINE__);

    uint32_t h_error_count = 0;
    double h_mse_sum;
    cudaMemcpy(&h_error_count, d_error_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mse_sum, d_mse_sum, sizeof(double), cudaMemcpyDeviceToHost);
    CheckCuda(__LINE__);
    double mse = h_mse_sum / coordinate_num;
    double psnr = 20.0 * log10(max_val - min_val) - 10.0 * log10(mse);

    printf("[%s] Error count = %u / %zu\n", label, h_error_count, coordinate_num);
    printf("[%s] MSE = %e\n", label, mse);
    printf("[%s] PSNR = %.4f dB\n", label, psnr);

    cudaFree(d_error_count);
    cudaFree(d_mse_sum);

    return {h_error_count, psnr};
}

std::pair<uint32_t, float> verify_and_psnr(
        double *h_dec_data, double *h_ref_data,
        uint64_t coordinate_num, double eb,
        double min_val, double max_val,
        const char *label) {
    thrust::device_vector<double> d_dec(h_dec_data, h_dec_data + coordinate_num);
    thrust::device_vector<double> d_ref(h_ref_data, h_ref_data + coordinate_num);
    CheckCuda(__LINE__);
    thrust::sort(d_dec.begin(), d_dec.end());
    thrust::sort(d_ref.begin(), d_ref.end());
    CheckCuda(__LINE__);
    uint32_t *d_error_count;
    double *d_mse_sum;
    cudaMalloc(&d_error_count, sizeof(uint32_t));
    cudaMalloc(&d_mse_sum, sizeof(double));
    cudaMemset(d_error_count, 0, sizeof(uint32_t));
    cudaMemset(d_mse_sum, 0, sizeof(double));
    CheckCuda(__LINE__);
    int32_t threads = 256;
    uint64_t blocks = (coordinate_num + threads - 1) / threads;
    compute_error_and_mse_double<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_dec.data()),
            thrust::raw_pointer_cast(d_ref.data()),
            coordinate_num, eb,
            d_error_count, d_mse_sum);
    CheckCuda(__LINE__);

    uint32_t h_error_count = 0;
    double h_mse_sum;
    cudaMemcpy(&h_error_count, d_error_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mse_sum, d_mse_sum, sizeof(double), cudaMemcpyDeviceToHost);
    CheckCuda(__LINE__);
    double mse = h_mse_sum / coordinate_num;
    double psnr = 20.0 * log10(max_val - min_val) - 10.0 * log10(mse);

    printf("[%s] Error count = %u / %zu\n", label, h_error_count, coordinate_num);
    printf("[%s] MSE = %e\n", label, mse);
    printf("[%s] PSNR = %.4f dB\n", label, psnr);

    cudaFree(d_error_count);
    cudaFree(d_mse_sum);

    return {h_error_count, psnr};
}


double combine_psnr(const std::vector<double> &psnr_list) {
    if (psnr_list.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> nrmse_list;
    nrmse_list.reserve(psnr_list.size());

    for (double psnr: psnr_list) {
        double nrmse = std::pow(10.0, psnr / -20.0);
        nrmse_list.push_back(nrmse);
    }

    double sum_sq = 0.0;
    for (double nrmse: nrmse_list) {
        sum_sq += nrmse * nrmse;
    }

    double rms_nrmse = std::sqrt(sum_sq / nrmse_list.size());

    if (rms_nrmse > 0.0) {
        return -20.0 * std::log10(rms_nrmse);
    } else {
        return std::numeric_limits<double>::infinity();
    }
}



void verifyCUDA(float *h_dec_data_x, float *h_dec_data_y, float *h_dec_data_z, float *h_x, float *h_y, float *h_z,
                uint64_t coordinate_num, float *min_max_values, float eb, uint32_t dim) {
    printf("verify by CUDA...\n");
    auto [err_count_x, psnr_x] = verify_and_psnr(h_dec_data_x, h_x, coordinate_num, eb, min_max_values[0],
                                                 min_max_values[3], "X");
    auto [err_count_y, psnr_y] = verify_and_psnr(h_dec_data_y, h_y, coordinate_num, eb, min_max_values[1],
                                                 min_max_values[4], "Y");

    std::vector<double> psnrs = {psnr_x, psnr_y};
    uint64_t total_error_count = err_count_x + err_count_y;
    if (dim == 3) {
        auto [err_count_z, psnr_z] = verify_and_psnr(h_dec_data_z, h_z, coordinate_num, eb, min_max_values[2],
                                                     min_max_values[5], "Z");
        psnrs = {psnr_x, psnr_y, psnr_z};
        total_error_count = err_count_x + err_count_y + err_count_z;
    }

    double psnr = combine_psnr(psnrs);

    printf("Error Count: %zu\n", total_error_count);
    printf("PSNR: %f\n", psnr);
}

void verifyCUDA(double *h_dec_data_x, double *h_dec_data_y, double *h_dec_data_z, double *h_x, double *h_y, double *h_z,
                uint64_t coordinate_num, double *min_max_values, double eb, uint32_t dim) {
    printf("verify by CUDA...\n");
    auto [err_count_x, psnr_x] = verify_and_psnr(h_dec_data_x, h_x, coordinate_num, eb, min_max_values[0],
                                                 min_max_values[3], "X");
    auto [err_count_y, psnr_y] = verify_and_psnr(h_dec_data_y, h_y, coordinate_num, eb, min_max_values[1],
                                                 min_max_values[4], "Y");

    std::vector<double> psnrs = {psnr_x, psnr_y};
    uint64_t total_error_count = err_count_x + err_count_y;
    if (dim == 3) {
        auto [err_count_z, psnr_z] = verify_and_psnr(h_dec_data_z, h_z, coordinate_num, eb, min_max_values[2],
                                                     min_max_values[5], "Z");
        psnrs = {psnr_x, psnr_y, psnr_z};
        total_error_count = err_count_x + err_count_y + err_count_z;
    }

    double psnr = combine_psnr(psnrs);

    printf("Error Count: %zu\n", total_error_count);
    printf("PSNR: %f\n", psnr);
}


float *readSingleFrame(const std::string &filename, uint64_t numPoints, uint64_t timeFrameIndex) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    // Calculate the number of points per frame
    uint64_t pointsPerFrame = numPoints * 3;
    uint64_t frameSize = pointsPerFrame * sizeof(float);

    // Seek to the beginning of the desired frame
    file.seekg(frameSize * timeFrameIndex, std::ios::beg);

    // Allocate memory for one frame of data
    float *h_xyz = new float[pointsPerFrame];

    // Read one frame of data
    file.read(reinterpret_cast<char *>(h_xyz), frameSize);

    if (!file) {
        std::cerr << "Error reading file: only " << file.gcount() << " bytes could be read" << std::endl;
        delete[] h_xyz;
        return nullptr;
    }

    file.close();
    return h_xyz;
}

float *loadF32File(const std::string &filename, uint64_t &numFloats) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "unable to open: " << filename << std::endl;
        return nullptr;
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    numFloats = fileSize / sizeof(float);

    float *data = new float[numFloats];

    if (file.read(reinterpret_cast<char *>(data), fileSize)) {
        std::cout << "loaded " << numFloats << " float " << std::endl;
    } else {
        std::cerr << "failed" << std::endl;
        delete[] data;
        return nullptr;
    }

    file.close();
    return data;
}

void printGPUMemoUsage(int32_t num) {
    printf("%d \n", num);
    // check supported device
    int32_t deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "no cuda device" << std::endl;
        return;
    }
    // get current device
    int32_t device_id = 0;
    cudaGetDevice(&device_id);

    // get gpu mem info
    uint64_t free_memory = 0;
    uint64_t total_memory = 0;
    cudaError_t cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Error: cudaMemGetInfo failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return;
    }

    // convert to mb
    double total_memory_mb = static_cast<double>(total_memory) / (1024.0 * 1024.0);
    double free_memory_mb = static_cast<double>(free_memory) / (1024.0 * 1024.0);
    double used_memory_mb = total_memory_mb - free_memory_mb;

    std::cout << "GPU ID: " << device_id << std::endl;
    std::cout << "total: " << total_memory_mb << " MB" << std::endl;
    std::cout << "used: " << used_memory_mb << " MB" << std::endl;
    std::cout << "remaining: " << free_memory_mb << " MB" << std::endl;
}

void write_unsigned_char_to_file(const unsigned char *h_encoded_bids, uint64_t length, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("unable to open\n");
        return;
    }

    uint64_t elements_written = fwrite(h_encoded_bids, sizeof(unsigned char), length, file);
    if (elements_written != length) {
        printf("write error\n");
    }

    fclose(file);
}

void write_uint64_t_to_file(const uint64_t *h_encoded_bids, uint64_t length, const char *filename) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        printf("unable to open\n");
        return;
    }

    uint64_t elements_written = fwrite(h_encoded_bids, sizeof(uint64_t), length, file);
    if (elements_written != length) {
        printf("write error\n");
    }

    fclose(file);
}

void write_int_to_file(const int32_t *h_encoded_bids, uint64_t length, const char *filename) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        printf("unable to open\n");
        return;
    }

    uint64_t elements_written = fwrite(h_encoded_bids, sizeof(int32_t), length, file);
    if (elements_written != length) {
        printf("write error\n");
    }

    fclose(file);
}

void load_file_to_cuda_hostmem(const char *filename, unsigned char **host_ptr, uint64_t *file_size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        throw std::runtime_error("Failed to open input file");
    }

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    rewind(file);

//    *host_ptr = (unsigned char*)malloc(*file_size * sizeof(unsigned char));

    cudaError_t err = cudaMallocHost(host_ptr, *file_size);
    if (err != cudaSuccess) {
        fclose(file);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    uint64_t read = fread(*host_ptr, sizeof(unsigned char), *file_size, file);
    if (read != *file_size) {
        cudaFree(*host_ptr);
        fclose(file);
        throw std::runtime_error("Failed to read complete data");
    }

    fclose(file);
}

//void write_xyz_to_aos_file(const float *h_dec_data_x, const float *h_dec_data_y, const float *h_dec_data_z, uint64_t N,
//                           const std::string &filename, uint32_t dim) {
//    std::ofstream fout(filename, std::ios::binary);
//    if (!fout) {
//        throw std::runtime_error("Failed to open file for writing");
//    }
//    if (dim == 3) {
//        std::vector<float> interleaved(3 * N);
//        for (uint64_t i = 0; i < N; ++i) {
//            interleaved[3 * i + 0] = h_dec_data_x[i];
//            interleaved[3 * i + 1] = h_dec_data_y[i];
//            interleaved[3 * i + 2] = h_dec_data_z[i];
//        }
//        fout.write(reinterpret_cast<const char *>(interleaved.data()), interleaved.size() * sizeof(float));
//    }
//
//    if (dim == 2){
//        std::vector<float> interleaved(2 * N);
//        for (uint64_t i = 0; i < N; ++i) {
//            interleaved[2 * i + 0] = h_dec_data_x[i];
//            interleaved[2 * i + 1] = h_dec_data_y[i];
//        }
//        fout.write(reinterpret_cast<const char *>(interleaved.data()), interleaved.size() * sizeof(float));
//    }
//    fout.close();
//}


void
verify(float *h_dec_data_x, float *h_dec_data_y, float *h_dec_data_z, float *h_x, float *h_y, float *h_z,
       uint64_t numPoints, float *min_max_values, uint64_t coordinate_num, float eb) {
    printf("sorting decompressed data...\n");
    std::sort(h_dec_data_x, h_dec_data_x + numPoints);
    std::sort(h_dec_data_y, h_dec_data_y + numPoints);
    std::sort(h_dec_data_z, h_dec_data_z + numPoints);

    printf("sorting original data...\n");
    std::sort(h_x, h_x + numPoints);
    std::sort(h_y, h_y + numPoints);
    std::sort(h_z, h_z + numPoints);

    uint64_t big_error_count_t = 0;
    double error_sum_x = 0;
    double error_sum_y = 0;
    double error_sum_z = 0;
    uint64_t temp_count = 0;
    printf("start to verify...\n");
    for (uint64_t i = 0; i < numPoints; i++) {
        float err1 = fabs(h_x[i] - h_dec_data_x[i]);
        float err2 = fabs(h_y[i] - h_dec_data_y[i]);
        float err3 = fabs(h_z[i] - h_dec_data_z[i]);

        error_sum_x += err1 * err1;
        error_sum_y += err2 * err2;
        error_sum_z += err3 * err3;
        temp_count += 3;
        if (err1 > 1.01 * eb) {
            big_error_count_t++;
            printf("Error: x[%zu]: %f, recovered_xyz[%zu]: %f\n", i, h_x[i], i,
                   h_dec_data_x[i]);
        }
        if (err2 > 1.01 * eb) {
            big_error_count_t++;
            printf("Error: y[%zu]: %f, recovered_xyz[%zu]: %f\n", i, h_y[i], i,
                   h_dec_data_y[i]);
        }
        if (err3 > 1.01 * eb) {
            big_error_count_t++;
            printf("Error: z[%zu]: %f, recovered_xyz[%zu]: %f\n", i, h_z[i], i,
                   h_dec_data_z[i]);
        }
    }
    assert (temp_count == coordinate_num * 3);

    double mse_x = error_sum_x / coordinate_num;
    double mse_y = error_sum_y / coordinate_num;
    double mse_z = error_sum_z / coordinate_num;
    printf("MSE_x: %f\n", mse_x);
    printf("MSE_y: %f\n", mse_y);
    printf("MSE_z: %f\n", mse_z);
    double psnr_x = 20 * log10(min_max_values[3] - min_max_values[0]) - 10 * log10(mse_x);
    double psnr_y = 20 * log10(min_max_values[4] - min_max_values[1]) - 10 * log10(mse_y);
    double psnr_z = 20 * log10(min_max_values[5] - min_max_values[2]) - 10 * log10(mse_z);
    printf("Max range_x: %f\n", min_max_values[3] - min_max_values[0]);
    printf("Max range_y: %f\n", min_max_values[4] - min_max_values[1]);
    printf("Max range_z: %f\n", min_max_values[5] - min_max_values[2]);
    printf("PSNR_x: %f\n", psnr_x);
    printf("PSNR_y: %f\n", psnr_y);
    printf("PSNR_z: %f\n", psnr_z);
    std::vector<double> psnrs = {psnr_x, psnr_y, psnr_z};
    double psnr = combine_psnr(psnrs);
    printf("PSNR: %f\n", psnr);
    printf("Error count: %zu\n", big_error_count_t);
    printf("Total number: %zu\n", temp_count);
}
