#include <random>

#include "knn_search.cuh"

FastGicp::SortingKernel::SortingKernel(int k, thrust::device_vector<thrust::pair<double, int>>& kNeighbors)
    : k_(k), kNeighborsPtr_(kNeighbors.data())
{
}

__host__ __device__ void FastGicp::SortingKernel::operator()(int idx) const
{
    thrust::device_ptr<thrust::pair<double, int>> neighbors2SortPtr = kNeighborsPtr_ + idx * k_;

    struct Cmp {
        __host__ __device__ bool operator()(const thrust::pair<double, int>& lhs, const thrust::pair<double, int>& rhs)
        {
            return lhs.first < rhs.first; // Sort elements in the ascending order
        }
    };

    thrust::sort(thrust::device, neighbors2SortPtr, neighbors2SortPtr + k_, Cmp());
}

FastGicp::NeighborSearchKernel::NeighborSearchKernel(int k, const thrust::device_vector<Eigen::Vector3d>& target,
    thrust::device_vector<thrust::pair<double, int>>& kNeighbors)
    : k_(k), numTargetPts_(target.size()), targetConstPtsPtr_(target.data()), kNeighborsPtr_(kNeighbors.data())
{
}

template <typename Tuple>
__host__ __device__ void FastGicp::NeighborSearchKernel::operator()(const Tuple& indexedPts) const
{
    // threadIdx doesn't work because thrust split for_each in two loops
    int idx = thrust::get<0>(indexedPts);
    const Eigen::Vector3d& pt = thrust::get<1>(indexedPts);

    const Eigen::Vector3d* const tarPts = thrust::raw_pointer_cast(targetConstPtsPtr_);
    thrust::pair<double, int>* kNeighbors = thrust::raw_pointer_cast(kNeighborsPtr_) + idx * k_;

    struct Cmp {
        __host__ __device__ bool operator()(const thrust::pair<double, int>& lhs, const thrust::pair<double, int>& rhs)
        {
            return lhs.first < rhs.first; // Sort elements in the ascending order
        }
    };

    for (int i = 0; i < k_; i++) {
        double sqDist = (tarPts[i] - pt).squaredNorm();
        kNeighbors[i] = thrust::make_pair(sqDist, i);
    }
    thrust::sort(thrust::device, kNeighbors, kNeighbors + k_, Cmp());

    for (int i = k_; i < numTargetPts_; i++) {
        double sqDist = (tarPts[i] - pt).squaredNorm();

        // 1: k-1 to locate the last element
        if (sqDist < kNeighbors[k_ - 1].first) {
            kNeighbors[k_ - 1] = thrust::make_pair(sqDist, i);
            thrust::sort(thrust::device, kNeighbors, kNeighbors + k_, Cmp());
        }
    }
}

void FastGicp::KnnSearch::SearchNeighbors(const thrust::device_vector<Eigen::Vector3d>& srcPts,
    const thrust::device_vector<Eigen::Vector3d>& tarPts, const int k, const bool doSort,
    thrust::device_vector<thrust::pair<double, int>>& kNeighbors)
{
    thrust::device_vector<int> dIndices(srcPts.size());
    thrust::sequence(dIndices.begin(), dIndices.end());

    auto first = thrust::make_zip_iterator(thrust::make_tuple(dIndices.begin(), srcPts.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(dIndices.end(), srcPts.end()));

    kNeighbors.resize(srcPts.size() * k, thrust::make_pair(-1.0f, -1));
    thrust::for_each(first, last, NeighborSearchKernel(k, tarPts, kNeighbors));

    if (doSort) {
        thrust::for_each(dIndices.begin(), dIndices.end(), SortingKernel(k, kNeighbors));
    }
}
