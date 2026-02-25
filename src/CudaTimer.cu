#include "cudaTimer.cuh"


CudaTimer::CudaTimer() : elapsedTime(0.0f) {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void CudaTimer::start() {
    cudaEventRecord(startEvent, 0);
}

void CudaTimer::stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
}

float CudaTimer::elapsedMilliseconds() const {
    return elapsedTime;
}

