#include "cov_estimation.cuh"

FastGicp::CovEstimationKernel::CovEstimationKernel(const int k, const thrust::device_vector<Eigen::Vector3d>& pts,
    const thrust::device_vector<int>& neighborsIndices, thrust::device_vector<Eigen::Matrix3d>& covs)
    : k_(k), ptsPtr_(pts.data()), neighborsIndicesPtr_(neighborsIndices.data()), covsPtr_(covs.data())
{
}

__host__ __device__ void FastGicp::CovEstimationKernel::operator()(const int idx) const
{
    const Eigen::Vector3d* const ptsPtr = thrust::raw_pointer_cast(ptsPtr_);
    const int* const neighborsIndicesPtr = thrust::raw_pointer_cast(neighborsIndicesPtr_) + idx * k_;
    Eigen::Matrix3d* const covPtr = thrust::raw_pointer_cast(covsPtr_) + idx;

    Eigen::Vector3d mean(0.0, 0.0, 0.0);
    Eigen::Matrix3d cov;
    cov.setZero();
    for (int i = 0; i < k_; ++i) {
        const auto neighborIdx = neighborsIndicesPtr[i];
        const auto& pt = ptsPtr[neighborIdx];
        mean += pt;
        (*covPtr) += pt * pt.transpose();
    }
    mean /= k_;
    (*covPtr) = (*covPtr) / k_;
    (*covPtr) -= mean * mean.transpose();
}

__host__ __device__ void FastGicp::RegularizationSvdKernel::operator()(Eigen::Matrix3d& cov) const
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(cov);
    Eigen::Matrix3d values = Eigen::Vector3d(1e-3, 1, 1).asDiagonal();
    Eigen::Matrix3d vectors = eig.eigenvectors();
    Eigen::Matrix3d vectorsInv = vectors.transpose();
    // here: vectors.transpose() == vectors.inverse(). But why vector.inverse() doesn't work???
    // Eigen::Matrix3d vectorsInv = vectors.inverse();

    cov = vectors * values * vectorsInv;
}

void FastGicp::CovEstimation::EstimateCov(const int k, const thrust::device_vector<Eigen::Vector3d>& pts,
    const thrust::device_vector<int>& neighborsIndices, thrust::device_vector<Eigen::Matrix3d>& covs)
{
    thrust::device_vector<int> indices(pts.size());
    thrust::sequence(indices.begin(), indices.end());
    covs.resize(pts.size());

    thrust::for_each(indices.begin(), indices.end(), CovEstimationKernel(k, pts, neighborsIndices, covs));
    thrust::for_each(covs.begin(), covs.end(), RegularizationSvdKernel());
}