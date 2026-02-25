#pragma once
#include <string>
#include <cstdint>
#include <cudaCompress.cuh>

class ICudaCompressor {
public:
    virtual ~ICudaCompressor() = default;

    virtual void compress(void *d_xyz, unsigned char *comp_data, uint64_t *comp_data_size,
                          const std::string &comp_type) = 0;

    virtual void decompress(void *decData_x, void *decData_y, void *decData_z, unsigned char *d_compressed_data,
                    const std::string &nvcomp_type) = 0;

    virtual const CudaConfig& getConfig() const = 0;
};