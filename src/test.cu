#include "test.cuh"

struct UntiePairSecond {
    __host__ __device__ int operator()(thrust::pair<double, int>& p) const
    {
        return p.second;
    }
};

void Test::KnnSearchTest::Test()
{
    thrust::host_vector<Eigen::Vector3d> hSrcPts;
    thrust::host_vector<Eigen::Vector3d> hTarPts;
    std::default_random_engine randomEngine;
    std::uniform_real_distribution<double> dist(0, 10.0);

    const size_t NUM_SRC_SIZE = 100;
    const size_t NUM_TAR_SIZE = 100;
    const size_t K = 10;

    for (int i = 0; i < NUM_SRC_SIZE; ++i) {
        Eigen::Vector3d pt{dist(randomEngine), dist(randomEngine), dist(randomEngine)};
        hSrcPts.push_back(pt);
    }

    for (int i = 0; i < NUM_TAR_SIZE; ++i) {
        Eigen::Vector3d pt{dist(randomEngine), dist(randomEngine), dist(randomEngine)};
        hTarPts.push_back(pt);
    }

    thrust::device_vector<Eigen::Vector3d> dSrcPts(hSrcPts);
    thrust::device_vector<Eigen::Vector3d> dTarPts(hTarPts);
    thrust::device_vector<thrust::pair<double, int>> kNeighbors;

    FastGicp::KnnSearch ks;
    ks.SearchNeighbors(dSrcPts, dTarPts, K, false, kNeighbors);

    // std::cin.get();

    // for (int i = 0; i < kNeighbors.size(); ++i) {
    //     auto n = static_cast<thrust::pair<double, int>>(kNeighbors[i]);
    //     std::cout << i << " " << n.first << " " << n.second << std::endl;
    // }
}

void Test::CovEstimationTest::Test()
{
    thrust::host_vector<Eigen::Vector3d> hPts;
    std::default_random_engine randomEngine;
    std::uniform_real_distribution<double> dist(0, 10.0);

    const size_t NUM_PTS_SIZE = 100;
    const size_t K = 10;

    for (int i = 0; i < NUM_PTS_SIZE; ++i) {
        Eigen::Vector3d pt{dist(randomEngine), dist(randomEngine), 0};
        hPts.push_back(pt);
    }

    thrust::device_vector<Eigen::Vector3d> dPts(hPts);
    thrust::device_vector<thrust::pair<double, int>> kNeighbors;

    FastGicp::KnnSearch ks;
    ks.SearchNeighbors(hPts, hPts, K, false, kNeighbors);

    LOG(INFO) << "Points size: " << NUM_PTS_SIZE << "; "
              << "Neighbors size: " << kNeighbors.size() << "; ";

    thrust::device_vector<int> neighborIndices;
    neighborIndices.resize(kNeighbors.size());

    thrust::transform(kNeighbors.begin(), kNeighbors.end(), neighborIndices.begin(), UntiePairSecond());
    thrust::device_vector<Eigen::Matrix3d> covs;

    LOG(INFO) << "Start cet";

    FastGicp::CovEstimation ce;
    ce.EstimateCov(K, dPts, neighborIndices, covs);

    for (int i = 0; i < covs.size(); ++i) {
        auto cov = static_cast<Eigen::Matrix3d>(covs[i]);
        LOG(INFO) << "Number: " << i << "\n" << cov;
    }

    // std::cin.get();

    // for (int i = 0; i < kNeighbors.size(); ++i) {
    //     auto n = static_cast<thrust::pair<double, int>>(kNeighbors[i]);
    //     std::cout << i << " " << n.first << " " << n.second << std::endl;
    // }
}