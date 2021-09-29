#ifndef INCLUDE_COV_ESTIMATION_CUH
#define INCLUDE_COV_ESTIMATION_CUH

#include <cublas_v2.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace FastGicp {
namespace {
struct CovEstimationKernel {
public:
    CovEstimationKernel(const int k, const thrust::device_vector<Eigen::Vector3d>& pts,
        const thrust::device_vector<int>& neighborsIndices, thrust::device_vector<Eigen::Matrix3d>& covs);
    __host__ __device__ void operator()(const int idx) const;

private:
    const int k_;
    thrust::device_ptr<const Eigen::Vector3d> ptsPtr_;
    thrust::device_ptr<Eigen::Matrix3d> covsPtr_;
    thrust::device_ptr<const int> neighborsIndicesPtr_;
};

struct RegularizationSvdKernel {
public:
    RegularizationSvdKernel() = default;
    __host__ __device__ void operator()(Eigen::Matrix3d& cov) const;
};
} // namespace

class CovEstimation {
public:
    CovEstimation() = default;
    ~CovEstimation() = default;
    void EstimateCov(const int k, const thrust::device_vector<Eigen::Vector3d>& pts,
        const thrust::device_vector<int>& neighborsIndices, thrust::device_vector<Eigen::Matrix3d>& covs);
};

} // namespace FastGicp

#endif // INCLUDE_COV_ESTIMATION_CUH