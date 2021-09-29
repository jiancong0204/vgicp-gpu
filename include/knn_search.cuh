#ifndef INCLUDE_KNN_SEARCH_CUH
#define INCLUDE_KNN_SEARCH_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <Eigen/Core>

namespace FastGicp {
namespace {
struct SortingKernel {
public:
    SortingKernel(int k, thrust::device_vector<thrust::pair<double, int>>& kNeighbors);
    __host__ __device__ void operator()(const int idx) const;

private:
    const int k_;
    thrust::device_ptr<thrust::pair<double, int>> kNeighborsPtr_;
};

struct NeighborSearchKernel {
public:
    NeighborSearchKernel(int k, const thrust::device_vector<Eigen::Vector3d>& target,
        thrust::device_vector<thrust::pair<double, int>>& kNeighbors);
    template <typename Tuple> __host__ __device__ void operator()(const Tuple& idx_x) const;

private:
    const int k_;
    const int numTargetPts_;
    thrust::device_ptr<const Eigen::Vector3d> targetConstPtsPtr_;
    thrust::device_ptr<thrust::pair<double, int>> kNeighborsPtr_;
};
} // namespace

class KnnSearch {
public:
    KnnSearch() = default;
    ~KnnSearch() = default;
    void SearchNeighbors(const thrust::device_vector<Eigen::Vector3d>& srcPts,
        const thrust::device_vector<Eigen::Vector3d>& tarPts, const int k, const bool doSort,
        thrust::device_vector<thrust::pair<double, int>>& kNeighbors);
};
} // namespace FastGicp

#endif // INCLUDE_KNN_SEARCH_CUH