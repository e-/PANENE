#include <iostream>
#include <vector>
#include <fstream>

#include <dist.h>
#include <util/result_set.h>
#include <data_source/binary_data_source.h>
#include <cstdlib>

using namespace panene;
using Source = BinaryDataSource<size_t, L2<float>>;

inline float distance(const Source &a, const size_t i,
    const Source &b, const size_t j) {
    int d = a.dim();
    float sum = 0;
    for (int k = 0; k < d; ++k) {
        float x = a.get(i, k);
        float y = b.get(j, k);
        sum += (x - y) * (x - y);
    }
    return sqrt(sum);
}

void getExactNN(const Source &train,
    const Source &test,
    std::vector<ResultSet<Source::IDType, Source::DistanceType>> &exactResults,
    size_t k) {

    size_t trainN = train.size();
    size_t d = train.dim();
    size_t testN = test.size();
    size_t count = 0;

#pragma omp parallel for
    for (int q = 0; q < (int)testN; q++) {
        count++;
        if (count % 500 == 0)
            std::cerr << count << " ";
        auto &result = exactResults[q];

        for (size_t i = 0; i < trainN; i++) {
            float dist = distance(train, i, test, q);
            if (!result.full() || result.worstDist > dist) {
                result << Neighbor<Source::IDType, Source::DistanceType>(i, dist);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 8) {
        std::cout << "compute the exact neighbors of query points from train points" << std::endl;
        std::cout << argv[0] << " <trainPath> <queryPath> <answerPath> <trainN> <testN> <dim> <k>" << std::endl;
        return 1;
    }

    std::string trainPath = std::string(argv[1]);
    std::string testPath = std::string(argv[2]);
    std::string answerPath = std::string(argv[3]);
    size_t trainN = std::atoi(argv[4]);
    size_t testN = std::atoi(argv[5]);
    size_t dim = std::atoi(argv[6]);
    size_t k = std::atoi(argv[7]);

    std::cout << "trainPath: " << trainPath << std::endl;
    std::cout << "testPath: " << testPath << std::endl;
    std::cout << "answerPath: " << answerPath << std::endl;
    std::cout << "trainN: " << trainN << std::endl;
    std::cout << "testN: " << testN << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "k: " << k << std::endl;

    Source trainSource;
    Source testSource;

    trainN = trainSource.open(trainPath, trainN, dim);
    testN = testSource.open(testPath, testN, dim);

    std::cout << "computing exact neighbors for test points (N = " << testN << ") from training points (N = " << trainN << ")" << std::endl;

    std::vector<ResultSet<Source::IDType, Source::DistanceType>> exactResults(testN);
    for (size_t i = 0; i < testN; ++i) {
        exactResults[i] = ResultSet<Source::IDType, Source::DistanceType>(k);
    }


    getExactNN(trainSource, testSource, exactResults, k);

    std::cout << "exact neighbors computation done" << std::endl;

    std::cout << "writing to " << answerPath << std::endl;

    std::ofstream out(answerPath, std::ios::out);

    out << k << std::endl;
    for (size_t i = 0; i < testN; ++i) {
        for (size_t j = 0; j < k; ++j) {
            out << exactResults[i][j].id << " " << exactResults[i][j].dist << " ";
        }
        out << std::endl;
    }

    out.close();

    return 0;
}
