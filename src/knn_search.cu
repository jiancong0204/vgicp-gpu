#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <random>

#include "knn_search.cuh"

KnnSearch::SortingKernel::SortingKernel(int k, thrust::device_vector<thrust::pair<double, int>>& kNeighbors)
    : k_(k), kNeighbors_(kNeighbors.data())
{
}

__host__ __device__ void KnnSearch::SortingKernel::operator()(int idx) const
{
    thrust::device_ptr<thrust::pair<double, int>> neighbors2Sort = kNeighbors_ + idx * k_;

    struct Cmp {
        __host__ __device__ bool operator()(const thrust::pair<double, int>& lhs, const thrust::pair<double, int>& rhs)
        {
            return lhs.first < rhs.first; // Sort so that distances are increasing.
        }
    };

    thrust::sort(thrust::device, neighbors2Sort, neighbors2Sort + k_, Cmp());
}

KnnSearch::NeighborSearchKernel::NeighborSearchKernel(int k, const thrust::device_vector<Eigen::Vector3f>& target,
    thrust::device_vector<thrust::pair<double, int>>& k_neighbors)
    : k(k), num_target_points(target.size()), target_points_ptr(target.data()), k_neighbors_ptr(k_neighbors.data())
{
}

template <typename Tuple> __host__ __device__ void KnnSearch::NeighborSearchKernel::operator()(const Tuple& idx_x) const
{
    // threadIdx doesn't work because thrust split for_each in two loops
    int idx = thrust::get<0>(idx_x);
    const Eigen::Vector3f& x = thrust::get<1>(idx_x);

    // target points buffer & nn output buffer
    const Eigen::Vector3f* pts = thrust::raw_pointer_cast(target_points_ptr);
    thrust::pair<double, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

    struct compare_type {
        bool operator()(const thrust::pair<double, int>& lhs, const thrust::pair<double, int>& rhs)
        {
            return lhs.first < rhs.first;
        }
    };

    for (int i = 0; i < k; i++) {
        double sq_dist = (pts[i] - x).squaredNorm();
        k_neighbors[i] = thrust::make_pair(sq_dist, i);
    }
    thrust::sort(k_neighbors, k_neighbors + k - 1, compare_type());

    for (int i = k; i < num_target_points; i++) {
        double sq_dist = (pts[i] - x).squaredNorm();
        if (sq_dist < k_neighbors[k - 1].first) {
            k_neighbors[k - 1] = thrust::make_pair(sq_dist, i);
            thrust::sort(k_neighbors, k_neighbors + k - 1, compare_type());
        }
    }
}

void KnnSearch::KnnSearch::Test()
{
    thrust::host_vector<thrust::pair<double, int>> hVec;
    std::default_random_engine randomEngine;
    std::uniform_real_distribution<double> dist(0, 10.0);

    for (int i = 0; i < 100; ++i) {
        auto val = dist(randomEngine);
        hVec.push_back(thrust::make_pair(val, i));
    }

    for (int i = 0; i < 100; ++i) {
        auto p = static_cast<thrust::pair<double, int>>(hVec[i]);
        std::cout << i << " " << p.first << " " << p.second << std::endl;
    }
    thrust::device_vector<thrust::pair<double, int>> dVec(hVec);

    thrust::device_vector<int> dIdx(10);
    thrust::sequence(dIdx.begin(), dIdx.end());

    thrust::for_each(thrust::device, dIdx.begin(), dIdx.end(), SortingKernel(10, dVec));
    cudaDeviceSynchronize();

    // thrust::device_ptr<thrust::pair<double, int>> k_neighbors_ptr(dVec.data());
    // for (auto idx : dIdx) {
    //     thrust::device_ptr<thrust::pair<double, int>> k_neighbors = k_neighbors_ptr + 10 * idx;
    //     thrust::sort(k_neighbors, k_neighbors + 10);
    // }

    for (int i = 0; i < 100; ++i) {
        auto p = static_cast<thrust::pair<double, int>>(dVec[i]);
        std::cout << i << " " << p.first << " " << p.second << std::endl;
    }
}