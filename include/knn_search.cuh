#ifndef FIRST_INCLUDE_KNN_SEARCH_H
#define FIRST_INCLUDE_KNN_SEARCH_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <Eigen/Core>

namespace KnnSearch {
namespace {

struct SortingKernel {
    SortingKernel(int k, thrust::device_vector<thrust::pair<double, int>>& kNeighbors);
    __host__ __device__ void operator()(int idx) const;

    const int k_;
    thrust::device_ptr<thrust::pair<double, int>> kNeighbors_;
};

struct NeighborSearchKernel {
    NeighborSearchKernel(int k, const thrust::device_vector<Eigen::Vector3f>& target,
        thrust::device_vector<thrust::pair<double, int>>& k_neighbors);
    template <typename Tuple> __host__ __device__ void operator()(const Tuple& idx_x) const;

    const int k;
    const int num_target_points;
    thrust::device_ptr<const Eigen::Vector3f> target_points_ptr;

    thrust::device_ptr<thrust::pair<double, int>> k_neighbors_ptr;
};

} // namespace

class KnnSearch {
public:
    KnnSearch() = default;
    ~KnnSearch() = default;
    void Test();
};
} // namespace KnnSearch

#endif // FIRST_INCLUDE_KNN_SEARCH_H