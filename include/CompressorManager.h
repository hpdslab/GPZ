#ifndef GPZ_COMPRESSORMANAGER_H
#define GPZ_COMPRESSORMANAGER_H


#include "cudaCompress.cuh"
#include "cudaWarpCompress.cuh"
#include "cudaWarpCompressDouble.cuh"

class CompressorManager {

public:
    CompressorManager();

    ~CompressorManager();

    void compressWarp(void *d_xyz, unsigned char *d_comp_data, uint64_t *comp_size, std::string nvcomp_type,
                      ICudaCompressor *compressor);


    void decompressLoadParas(uint32_t *dim, uint64_t *numPoints, uint32_t *float_bits, const std::string &nvcomp_type,
                             uint64_t nvcomp_num, uint32_t nvcomp_dim, uint32_t float_bits_num, unsigned char *h_compressed_data);

    void decompressWarp(void *d_dec_data_x, void *d_dec_data_y, void *d_dec_data_z,
                        unsigned char *d_compressed_data, ICudaCompressor *decompressor,
                        const std::string &nvcomp_type);

//    void verify(float *h_x, float *h_y, float *h_z);
};


#endif //GPZ_COMPRESSORMANAGER_H
