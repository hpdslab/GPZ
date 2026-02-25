#ifndef CUDA_WARP_CPMPRESS_DOUBLE_H
#define CUDA_WARP_CPMPRESS_DOUBLE_H

#ifdef GPZ_USE_NVCOMP
#include "nvcomp.hpp"
#endif
#include "ICudaCompressor.cuh"
#include <cstdio>
#include <utility>

class CudaWarpCompressDouble : public ICudaCompressor {
public:
    explicit CudaWarpCompressDouble(CudaConfig cuda_config);

    CudaWarpCompressDouble();

    ~CudaWarpCompressDouble() override;

    void compress(void *d_xyz, unsigned char *comp_data, uint64_t *comp_data_size,
                  const std::string &comp_type) override;

    void decompress(void *decData_x, void *decData_y, void *decData_z, unsigned char *d_compressed_data,
                    const std::string &nvcomp_type) override;

    auto getConfig() const -> const CudaConfig & override {
        return cuda_config;
    }

private:
    CudaConfig cuda_config;

    void computeSegSize(double *d_x, double *d_y, double *d_z, double *d_cache, CudaConfig &cuda_config);

    void compressImpl(double *d_xyz, unsigned char *comp_data, uint64_t *comp_data_size,
                  const std::string &comp_type);

    void decompressImpl(double *decData_x, double *decData_y, double *decData_z, unsigned char *d_compressed_data,
                    const std::string &nvcomp_type);
};

#endif
