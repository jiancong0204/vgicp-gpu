#include <iostream>

#include "gpu.h"
#include "knn_search.cuh"

int main()
{
    printCudaVersion();
    KnnSearch::KnnSearch ks;
    ks.Test();
    return 0;
}
