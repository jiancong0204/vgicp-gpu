#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>

#include "gpu.h"
#include "test.cuh"

int main(int argc, char** argv)
{
    gflags::SetVersionString("0.0.1");
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    printCudaVersion();

    // Test::KnnSearchTest kst;
    // kst.Test();

    Test::CovEstimationTest cet;
    cet.Test();

    return 0;
}
