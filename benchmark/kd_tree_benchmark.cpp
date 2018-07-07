#include <benchmark/metadata.h>
#include <kd_tree_index.h>
#include <progressive_kd_tree_index.h>

using namespace panene;
typedef size_t IDType;
typedef float ElementType;
using Source = panene::BinaryDataSource<IDType, L2<ElementType>>;

void readAnswers(const std::string &path, size_t queryN, size_t &k, std::vector<std::vector<Neighbor<size_t, float>>> &neighbors) {
    std::ifstream infile;

    infile.open(path);

    if (!infile.is_open()) {
        std::cerr << "file " << path << " does not exist" << std::endl;
        throw;
    }

    infile >> k;

    std::cout << "K = " << k << std::endl;

    neighbors.resize(queryN);
    for (size_t i = 0; i < queryN; ++i) {
        neighbors[i].resize(k);

        for (size_t j = 0; j < k; ++j) {
            infile >> neighbors[i][j].id >> neighbors[i][j].dist;
        }
    }

    infile.close();
}

void measureMeanDistError(size_t queryN, size_t k, std::vector<std::vector<Neighbor<size_t, float>>> exactResults,
    std::vector<ResultSet<size_t, float>> results, float &meanDistError, float &accuracy) {

    meanDistError = 0;
    accuracy = 0;

    for (size_t i = 0; i < queryN; ++i) {
        meanDistError += results[i][k - 1].dist / exactResults[i][k - 1].dist;

        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < k; ++l) {
                if (results[i][j] == exactResults[i][l]) accuracy += 1;
            }
        }
    }

    meanDistError /= queryN;
    accuracy /= queryN * k;
}

void run(const char* base_) {
    const std::string base(base_);
    Timer timer;

    const size_t pointsN = 1000000;
    std::vector<Dataset> datasets = {
      Dataset(base, "glove", "shuffled", pointsN, 100)/*,
      Dataset(base, "glove", "original", pointsN, 100),
      Dataset(base, "sift", "shuffled", pointsN, 128),
      Dataset(base, "sift", "original", pointsN, 128)*/
    };

    const int maxRepeat = 1; //5;
    const int queryRepeat = 5;
    const IndexParams indexParam(4);
    const size_t maxIter = 2500; // if all data is read, it stops
    const size_t maxQueryN = 1000;

    SearchParams searchParam(2048);
    searchParam.cores = 8;

    std::fstream log;

#ifdef _WIN32
    log.open(base + "./log_ops.tsv", std::fstream::out);
#else
    log.open("./log.tsv", std::fstream::out);
#endif

    log <<
        "method\tdata\tversion\trepeat\tmaxOp\titer\tnumPointsInserted\timbalance1\timbalance2\timbalance3\timbalance4\tmax_depth\tQPS\tAccuracy\tmeanDistError\taddPointElapsed\tupdateIndexElapsed" << std::endl;

    size_t maxOps[] = { 1000, 3000, 5000, 10000 };
    size_t maxOpsN = sizeof(maxOps) / sizeof(size_t);

    float addPointWeights[] = { 0.35f }; // 0.2f, 0.35f, 0.5f };
    size_t weightN = sizeof(addPointWeights) / sizeof(float);

    int datasetIndex = -1;
    for (const auto& dataset : datasets) {
        datasetIndex++;

        Source trainDataSource(dataset.path);

        size_t trainN = trainDataSource.open(dataset.path, dataset.n, dataset.dim);

        // read query set
        Source queryDataSource(dataset.queryPath);

        size_t queryN = queryDataSource.open(dataset.queryPath, maxQueryN, dataset.dim);

        std::vector<std::vector<Source::ElementType>> queryPoints(queryN);

        for (size_t i = 0; i < queryN; ++i) {
            std::vector<ElementType> point(dataset.dim);

            for (size_t j = 0; j < dataset.dim; ++j) point[j] = queryDataSource.get(i, j);
            queryPoints[i] = point;
        }

        // read k and answers
        size_t k;
        std::vector<std::vector<Neighbor<IDType, ElementType>>> exactResults;

        readAnswers(dataset.answerPath, queryN, k, exactResults);

        std::cout << "testing dataset [" << dataset.name << "], loading completed" << std::endl;

        for (int repeat = 0; repeat < maxRepeat; ++repeat) {
            for (const size_t maxOp : maxOps) {

                // online first
                if (true) {
                    KDTreeIndex<Source> onlineIndex(&trainDataSource, indexParam);

                    size_t numPointsInserted = 0;

                    for (int r = 0; r < maxIter; ++r) {
                        std::cout << "(" << datasetIndex << "/" << datasets.size()
                            << ") (online) (" << repeat << "/" << maxRepeat << ") (" << r
                            << "/" << maxIter << ")" << std::endl;

                        // update the index with the given number operations

                        std::cout << "onlineIndex.addPoints called" << std::endl;
                        timer.begin();
                        size_t addNewPointResult = onlineIndex.addPoints(maxOp);
                        std::cout << "onlineIndex.addPoints done" << std::endl;

                        double addNewPointElapsed = timer.end();

                        std::cerr << "time : " << addNewPointElapsed << std::endl;

                        if (addNewPointResult == 0) break;
                        numPointsInserted += addNewPointResult;

                        double updateIndexElapsed = 0;

                        double searchElapsed = 0;

                        std::vector<ResultSet<IDType, ElementType>> results(queryN);

                        for (int qr = 0; qr < queryRepeat; ++qr) {
                            for (size_t i = 0; i < queryN; ++i) {
                                results[i] = ResultSet<IDType, ElementType>(k);
                            }

                            std::cout << "onlineIndex.knnSearch called()" << std::endl;
                            timer.begin();
                            onlineIndex.knnSearch(queryPoints, results, k, searchParam);
                            searchElapsed += timer.end();
                            std::cout << "onlineIndex.knnSearch done " << searchElapsed << std::endl;
                        }

                        // check the result
                        float meanDistError = 0;
                        float accuracy = 0;

                        measureMeanDistError(queryN, k, exactResults, results, meanDistError, accuracy);

                        double qps = 1.0 / (searchElapsed / queryN / queryRepeat);
                        log << "online" << "\t";
                        log << dataset.name << "\t" << dataset.version << "\t" << repeat << "\t" << maxOp << "\t" << r << "\t" << numPointsInserted << "\t";

                        auto imbalances = onlineIndex.recomputeImbalances();
                        for (auto imbalance : imbalances) {
                            log << imbalance << "\t";
                        }

                        /*auto imbalances2 = onlineIndex.getCachedImbalances();
                        for (auto imbalance : imbalances2) {
                          std::cerr << imbalance << "\t";
                        }*/

                        log << onlineIndex.computeMaxDepth() << "\t" << qps << "\t" << accuracy << "\t" << meanDistError << "\t";
                        log << addNewPointElapsed << "\t" << updateIndexElapsed << std::endl;
                    }
                }

                // then, test progressive trees
                for (size_t w = 0; w < weightN; ++w) {
                    float addPointWeight = addPointWeights[w];
                    size_t numPointsInserted = 0;

                    ProgressiveKDTreeIndex<Source> progressiveIndex(&trainDataSource, indexParam, TreeWeight(addPointWeight, 1 - addPointWeight), 100.0f);

                    for (int r = 0; r < maxIter; ++r) {
                        std::cout << "(" << datasetIndex << "/" << datasets.size()
                            << ") (progressive" << w << "/" << weightN << ") (" << repeat
                            << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;

                        // update the index with the given number operations
                        auto updateResult = progressiveIndex.run(maxOp);
                        if (updateResult.addPointResult == 0)
                            break;

                        double searchElapsed = 0;

                        std::vector<ResultSet<IDType, ElementType>> results(queryN);

                        for (int qr = 0; qr < queryRepeat; ++qr) {
                            for (size_t i = 0; i < queryN; ++i) {
                                results[i] = ResultSet<IDType, ElementType>(k);
                            }

                            std::cout << "progressiveIndex.knnSearch called()" << std::endl;
                            timer.begin();
                            progressiveIndex.knnSearch(queryPoints, results, k, searchParam);
                            searchElapsed += timer.end();
                            std::cout << "progressiveIndex.knnSearch done" << std::endl;
                        }

                        // check the result
                        ElementType meanDistError = 0;
                        float accuracy = 0;

                        measureMeanDistError(queryN, k, exactResults, results, meanDistError, accuracy);

                        double qps = 1.0 / (searchElapsed / queryN / queryRepeat);
                        log << "progressive" << addPointWeight << "\t";
                        log << dataset.name << "\t" << dataset.version << "\t" << repeat << "\t" << maxOp << "\t" << r << "\t" << updateResult.numPointsInserted << "\t";
                        auto imbalances = progressiveIndex.recomputeImbalances();
                        for (auto imbalance : imbalances) {
                            log << imbalance << "\t";
                        }
                        log << progressiveIndex.computeMaxDepth() << "\t" << qps << "\t" << accuracy << "\t" << meanDistError << "\t";
                        log << updateResult.addPointElapsed << "\t" << updateResult.updateIndexElapsed << std::endl;

                        for (size_t i = 0; i < 4; ++i) {
                            std::cout << "tree" << i << " " << progressiveIndex.trees[i]->getCachedImbalance() << " " << progressiveIndex.trees[i]->size << std::endl;
                        }
                        std::cout << "updateCost: " << progressiveIndex.getSize() * std::log2(progressiveIndex.getSize()) << std::endl;
                        std::cout << "queryLoss: " << progressiveIndex.queryLoss << std::endl;
                        if (progressiveIndex.updateStatus == BuildingTree) {
                            std::cout << "building a tree backstage" << std::endl;
                            progressiveIndex.printBackstage();
                        }
                        if (progressiveIndex.updateStatus == InsertingPoints) {
                            progressiveIndex.printBackstage();
                        }
                    }
                }
            }
        }
    }

    log.close();
}

int main(int argc, const char **argv) {
#ifdef _WIN32
    run("D:\\G\\work\\panene\\PANENE\\data");
#else 
    if (argc < 2) {
        std::cout << argv[0] << " <dataset_base_path>" << std::endl;
        return 1;
    }

    run(argv[1]);
#endif
    return 0;
}
