#include "gpu.h"

__global__ 
void CudaKernelFunctions::ProcessInGpu(int* data, uint sizeX, uint sizeY)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    double tmp = double(x) / double(sizeX);
    int red = int(tmp * 255.99);
    tmp = double(y) / double(sizeY);
    int green = int(tmp * 255.99);
    int blue = 50;

    data[(y * sizeX + x) * 3]     = red;
    data[(y * sizeX + x) * 3 + 1] = green;
    data[(y * sizeX + x) * 3 + 2] = blue;
    return;
}

void CudaWrapper::printCudaVersion()
{
    std::cout << "CUDA Compiled version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

void CudaWrapper::Init(const int& x, const int& y)
{
    sizeX_ = x;
    sizeY_ = y;
    grid_ = dim3{sizeY_, 1, 1};
    thread_ = dim3{sizeX_, 1, 1};
}

void CudaWrapper::Run()
{
    auto start = std::chrono::high_resolution_clock::now();
    // int* gpuImageData = nullptr;
    // cudaMalloc((void**)&gpuImageData, sizeX_ * sizeY_ * 3 * sizeof(int));
    // cudaMemset((void**)&gpuImageData, 0, sizeX_ * sizeY_ * 3 * sizeof(int));
    
    thrust::device_vector<int> gpuImageData(sizeX_ * sizeY_ * 3, 0);
    int* gpuImgPtr = thrust::raw_pointer_cast(gpuImageData.data());
    CudaKernelFunctions::ProcessInGpu<<<grid_, thread_>>>(gpuImgPtr, sizeX_, sizeY_);
    gpuImgPtr = nullptr;
    free(gpuImgPtr);
    // int* hostImageData = (int*)malloc(sizeX_ * sizeY_ * 3 * sizeof(int));
    // cudaMemcpy(hostImageData, gpuImageData, sizeX_ * sizeY_ * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    std::cout << ms_double.count() << std::endl;

    int* cpuImageData = (int*)malloc(sizeX_ * sizeY_ * 3 * sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < sizeY_; ++y) {
        for (int x = 0; x < sizeX_; ++x) {
            double tmp = double(x) / double(sizeX_);
            int red = int(tmp * 255.99);
            tmp = double(y) / double(sizeY_);
            int green = int(tmp * 255.99);
            int blue = 50;
            cpuImageData[(y * sizeX_ + x) * 3]     = red;
            cpuImageData[(y * sizeX_ + x) * 3 + 1] = green;
            cpuImageData[(y * sizeX_ + x) * 3 + 2] = blue;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    ms_double = end - start;
    std::cout << ms_double.count() << std::endl;

    std::ofstream outfile;
    outfile.open("IMG_GPU.ppm");
    outfile << "P3\n" << sizeX_ << " " << sizeY_ << "\n255\n";
    for (int j = sizeY_ - 1; j >= 0; --j) {
        for (int i = 0; i < sizeX_; ++i) {
            int ir = gpuImageData[j * sizeX_ * 3 + i * 3];
            int ig = gpuImageData[j * sizeX_ * 3 + i * 3 + 1];
            int ib = gpuImageData[j * sizeX_ * 3 + i * 3 + 2];
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    outfile.close();

    outfile.open("IMG_CPU.ppm");
    outfile << "P3\n" << sizeX_ << " " << sizeY_ << "\n255\n";
    for (int j = sizeY_ - 1; j >= 0; --j) {
        for (int i = 0; i < sizeX_; ++i) {
            int ir = cpuImageData[j * sizeX_ * 3 + i * 3];
            int ig = cpuImageData[j * sizeX_ * 3 + i * 3 + 1];
            int ib = cpuImageData[j * sizeX_ * 3 + i * 3 + 2];
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    outfile.close();

    // free(hostImageData);
    free(cpuImageData);
    // cudaFree(gpuImageData);
}