#ifndef INCLUDE_TEST_CUH
#define INCLUDE_TEST_CUH

#include <glog/logging.h>

#include "knn_search.cuh"
#include "cov_estimation.cuh"

namespace Test {
class KnnSearchTest {
public:
    KnnSearchTest() = default;
    ~KnnSearchTest() = default;
    void Test();
};

class CovEstimationTest {
public:
    CovEstimationTest() = default;
    ~CovEstimationTest() = default;
    void Test();
};
} // namespace Test

#endif // INCLUDE_TEST_CUH
