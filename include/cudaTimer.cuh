#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>

class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();

    void start();
    void stop();
    float elapsedMilliseconds() const;

private:
    cudaEvent_t startEvent, stopEvent;
    float elapsedTime;
};

#endif