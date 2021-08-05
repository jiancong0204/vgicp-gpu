#include <iostream>

#ifdef USE_CUDA
#include "gpu.h"
#endif

int main()
{
    int sizeX = 800;
    int sizeY = 400;

    CudaWrapper cw;
    cw.Init(sizeX, sizeY);
    cw.Run();
    return 0;
}
