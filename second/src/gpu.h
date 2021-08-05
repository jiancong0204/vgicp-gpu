#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// __global__ 
// void ProcessInGpu(int* data, uint sizeX, uint sizeY);
namespace CudaKernelFunctions {
    __global__
    void ProcessInGpu(int* data, uint sizeX, uint sizeY);
} // CudaKernelFunctions


class CudaWrapper {
public:
    CudaWrapper() = default;
    ~CudaWrapper() = default;
    void printCudaVersion();
    void Init(const int& x, const int& y);
    void Run();

private:
    dim3 grid_;
    dim3 thread_;
    uint sizeX_;
    uint sizeY_;
};