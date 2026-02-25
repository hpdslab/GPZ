#ifndef GPZ_CUDAUTILS_CUH
#define GPZ_CUDAUTILS_CUH
#include <cuda_runtime.h>

__inline__ __device__ double warpReduceMin(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ double warpReduceMax(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ float warpReduceMin(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ int warpReduceMax(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ float atomicMinFl(float *address, float val) {
    int *int_address = (int *) address;
    int old = *int_address;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(int_address, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__inline__ __device__ float atomicMaxFl(float *address, float val) {
    // convert the address to an integer
    int *int_address = (int *) address;
    int old = *int_address;
    int assumed;

    do {
        assumed = old;
        // compare and swap: if the current value is less than the target value, update it to the target value
        old = atomicCAS(int_address, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
        // if the value is modified by other threads, retry
    } while (assumed != old);
    return __int_as_float(old);
}

__inline__ __device__ int getEffectiveBits(uint32_t x) {
    return 32 - __clz(x);
}

__inline__ __device__ int getEffectiveBits(uint64_t x) {
    return 64 - __clzll(x);
}

__inline__ __device__ double atomicMinDbl(double *address, double val) {
    unsigned long long int *addr_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__inline__ __device__ double atomicMaxDbl(double* address, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

class cudaUtils {

};


#endif //GPZ_CUDAUTILS_CUH
