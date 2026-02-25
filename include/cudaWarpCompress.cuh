#ifndef CUDA_WARP_CPMPRESS_H
#define CUDA_WARP_CPMPRESS_H

#ifdef GPZ_USE_NVCOMP
#include "nvcomp.hpp"
#endif
#include "ICudaCompressor.cuh"
#include <cstdio>
#include <utility>

class CudaWarpCompress : public ICudaCompressor {
public:
    explicit CudaWarpCompress(CudaConfig cuda_config);

    CudaWarpCompress();

    ~CudaWarpCompress() override;

    void compress(void *d_xyz, unsigned char *comp_data, uint64_t *comp_data_size,
                  const std::string &comp_type) override;

    void decompress(void *decData_x, void *decData_y, void *decData_z, unsigned char *d_compressed_data,
                    const std::string &nvcomp_type) override;

    auto getConfig() const -> const CudaConfig & override {
        return cuda_config;
    }

private:
    CudaConfig cuda_config;

    void computeSegSize(float *d_x, float *d_y, float *d_z, float *d_cache, CudaConfig &cuda_config);

    void compressImpl(float *d_xyz, unsigned char *comp_data, uint64_t *comp_data_size,
                  const std::string &comp_type);

    void decompressImpl(float *decData_x, float *decData_y, float *decData_z, unsigned char *d_compressed_data,
                    const std::string &nvcomp_type);
};

#endif
