#include "utils.cuh"
#include "CompressorManager.h"
#include "cudaCompress.cuh"
#include "cudaWarpCompress.cuh"
#include "cudaTimer.cuh"
#include "cudaWarpCompressDouble.cuh"

CompressorManager::CompressorManager() = default;

CompressorManager::~CompressorManager() = default;

void CompressorManager::compressWarp(void *d_xyz, unsigned char *d_comp_buff, uint64_t *comp_size,
                                     const std::string nvcomp_type,
                                     ICudaCompressor *compressor) {
    compressor->compress(d_xyz, d_comp_buff, comp_size, nvcomp_type);
}

void
CompressorManager::decompressWarp(void *d_dec_data_x, void *d_dec_data_y, void *d_dec_data_z,
                                  unsigned char *d_compressed_data, ICudaCompressor *decompressor, const std::string &nvcomp_type) {
    decompressor->decompress(d_dec_data_x, d_dec_data_y, d_dec_data_z, d_compressed_data, nvcomp_type);
}

void CompressorManager::decompressLoadParas(uint32_t *dim, uint64_t *numPoints, uint32_t *float_bits, const std::string &nvcomp_type, uint64_t nvcomp_num, uint32_t nvcomp_dim, uint32_t float_bits_num, unsigned char *h_compressed_data) {
    if (!nvcomp_type.empty() && nvcomp_type != "none") {
        *dim = nvcomp_dim;
        *float_bits = float_bits_num;
        *numPoints = nvcomp_num;
    } else {
        memcpy(dim, h_compressed_data, sizeof(uint32_t));
        memcpy(float_bits, h_compressed_data + 12, sizeof(uint32_t));
        memcpy(numPoints, h_compressed_data + 4, sizeof(uint64_t));
    }
}




