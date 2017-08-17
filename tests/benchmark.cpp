#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>

#include <progressive_knn_table.h>
#include <kd_tree_index.h>
#include <binary_data_source.h>
#include <util/result_set.h>
#include <util/timer.h>
#include <util/matrix.h>

using namespace panene;

void readAnswers(const std::string &path, size_t queryN, size_t &k, std::vector<std::vector<Neighbor<size_t, float>>> &neighbors){
  std::ifstream infile;

  infile.open(path);

  if (!infile.is_open()) {
    std::cerr << "file " << path << " does not exist" << std::endl;
    throw;
  }

  infile >> k;
  
  std::cout << "K = " << k << std::endl;

  neighbors.resize(queryN);
  for(size_t i = 0; i < queryN; ++i) {
    neighbors[i].resize(k);

    for(size_t j = 0 ; j < k; ++j) {
      infile >> neighbors[i][j].id >> neighbors[i][j].dist;
    }
  }

  infile.close();
}

struct Dataset {
  std::string name;
  std::string version;
  std::string path;
  std::string queryPath;
  std::string answerPath;

  size_t n;
  size_t dim;

  Dataset() = default;
  Dataset(std::string name_, std::string version_, std::string path_, 
      std::string queryPath_, std::string answerPath_,
      size_t n_, size_t dim_) : name(name_), version(version_), path(path_), 
  queryPath(queryPath_), answerPath(answerPath_),
  n(n_), dim(dim_) {  }
};

// for test
#define BASE "D:\\G\\work\\panene\\PANENE\\data"

#define GLOVE_TRAIN_PATH(v) BASE "/glove/glove." #v ".bin"
#define GLOVE_QUERY_PATH BASE "/glove/test.bin"
#define GLOVE_ANSWER_PATH(v) BASE "/glove/glove." #v ".answer.txt"

#define SIFT_TRAIN_PATH(v) BASE "/sift/sift." #v ".bin"
#define SIFT_QUERY_PATH BASE "/sift/test.bin"
#define SIFT_ANSWER_PATH(v) BASE "/sift/sift." #v ".answer.txt"

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

void run() {
  Timer timer;

  const size_t pointsN = 1000000;
  std::vector<Dataset> datasets = {    
    /*Dataset("glove", "shuffled", 
        GLOVE_TRAIN_PATH(shuffled),
        GLOVE_QUERY_PATH,
        GLOVE_ANSWER_PATH(shuffled),
        pointsN, 100),*/
    Dataset("glove", "original", 
        GLOVE_TRAIN_PATH(original),
        GLOVE_QUERY_PATH,
        GLOVE_ANSWER_PATH(original),
        pointsN, 100),
    /*Dataset("sift", "shuffled", 
        SIFT_TRAIN_PATH(shuffled),
        SIFT_QUERY_PATH,
        SIFT_ANSWER_PATH(shuffled),
        pointsN, 128),*/
    /*Dataset("sift", "original", 
        SIFT_TRAIN_PATH(original),
        SIFT_QUERY_PATH,
        SIFT_ANSWER_PATH(original),
        pointsN, 128)*/
  };
  
  const int maxRepeat = 1; //5;
  const int queryRepeat = 5;
  const IndexParams indexParam(4);  
  const size_t maxIter = 1500; // if all data is read, it stops
  const size_t maxQueryN = 1000;

  SearchParams searchParam(4096);
  searchParam.cores = 1;

  std::fstream log;

#ifdef _WIN32
  log.open(BASE "./log.tsv", std::fstream::out);
#else
  log.open("./log.tsv", std::fstream::out);
#endif

  log <<
    "method\tdata\tversion\trepeat\tmaxOp\titer\tnumPointsInserted\timbalance1\timbalance2\timbalance3\timbalance4\tmax_depth\tQPS\tAccuracy\tmeanDistError\taddNewPointElapsed\tupdateIndexElapsed" << std::endl;

  float addPointWeights[] = { 0.1, 0.2, 0.3 }; // {0.25, 0.5, 0.75};
  size_t weightN = sizeof(addPointWeights) / sizeof(float);

  size_t maxOps[] = { 5000 };// , 10000 };
  size_t maxOpsN = sizeof(maxOps) / sizeof(size_t);

  for(const auto& dataset: datasets) {
    panene::BinaryDataSource trainDataSource(dataset.path);

    size_t trainN = trainDataSource.open(dataset.path, dataset.n, dataset.dim);
    
    // read query set
    panene::BinaryDataSource queryDataSource(dataset.queryPath);
    
    size_t queryN = queryDataSource.open(dataset.queryPath, maxQueryN, dataset.dim);

    std::vector<std::vector<float>> queryPoints(queryN);

    for(size_t i = 0; i < queryN; ++i) {
      std::vector<float> point(dataset.dim);

      for(size_t j = 0; j < dataset.dim; ++j) point[j] = queryDataSource.get(i, j);
      queryPoints[i] = point;
    }

    // read k and answers
    size_t k;
    std::vector<std::vector<Neighbor<size_t, float>>> exactResults;
    
    readAnswers(dataset.answerPath, queryN, k, exactResults);

    std::cout << "testing dataset [" << dataset.name << "], loading completed" << std::endl;

    for (int repeat = 0; repeat < maxRepeat; ++repeat) {
      for(const size_t maxOp: maxOps) {
       
        // online first
        {
          KDTreeIndex<L2<float>, BinaryDataSource> onlineIndex(indexParam);
          onlineIndex.setDataSource(&trainDataSource);
          
          size_t numPointsInserted = 0;       

          for (int r = 0; r < maxIter; ++r) {
            std::cout << "(online) (" << repeat << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;

            // update the index with the given number operations
            size_t addPointOps;

            std::cout << "onlineIndex.addPoints called" << std::endl;
            timer.begin();
            size_t addNewPointResult = onlineIndex.addPoints(maxOp);
            std::cout << "onlineIndex.addPoints done" << std::endl;

            double addNewPointElapsed = timer.end();
            if (addNewPointResult == 0) break;
            numPointsInserted += addNewPointResult;

            double updateIndexElapsed = 0;

            double searchElapsed = 0;

            std::vector<ResultSet<size_t, float>> results(queryN);

            for (int qr = 0; qr < queryRepeat; ++qr) {
              for (size_t i = 0; i < queryN; ++i) {
                results[i] = ResultSet<size_t, float>(k);
              }

              std::cout << "onlineIndex.knnSearch called()" << std::endl;
              timer.begin();
              onlineIndex.knnSearch(queryPoints, results, k, searchParam);
              searchElapsed += timer.end();
              std::cout << "onlineIndex.knnSearch done" << std::endl;
            }

            // check the result
            float meanDistError = 0;
            float accuracy = 0;

            measureMeanDistError(queryN, k, exactResults, results, meanDistError, accuracy);

            float qps = 1.0 / (searchElapsed / queryN / queryRepeat);
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
        for (int w = 0; w < weightN; ++w) {
          float addPointWeight = addPointWeights[w];
          size_t numPointsInserted = 0;

          ProgressiveKDTreeIndex<L2<float>, BinaryDataSource> progressiveIndex(indexParam);
          progressiveIndex.setDataSource(&trainDataSource);

          for (int r = 0; r < maxIter; ++r) {
            std::cout << "(" << w << "/" << weightN << ") (" << repeat << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;
            // update the index with the given number operations

            size_t addPointOps;

            if (progressiveIndex.updateStatus != progressiveIndex.NoUpdate) {
              addPointOps = maxOp * addPointWeight;
            }
            else {
              addPointOps = maxOp;
            }

            std::cout << "progressiveIndex.addPoints called() " << addPointOps << std::endl;
            timer.begin();
            size_t addNewPointResult = progressiveIndex.addPoints(addPointOps);
            double addNewPointElapsed = timer.end();
            std::cout << "progressiveIndex.addPoints done" << std::endl;

            if (addNewPointResult == 0) break;
            numPointsInserted += addNewPointResult;
            
            double updateIndexElapsed = 0;

            if (progressiveIndex.updateStatus != progressiveIndex.NoUpdate) {
              std::cout << "progressiveIndex.update called() " << maxOp - addNewPointResult << std::endl;
              timer.begin();
              progressiveIndex.update(maxOp - addNewPointResult);
              updateIndexElapsed = timer.end();
              std::cout << "progressiveIndex.update done" << std::endl;
            }

            double searchElapsed = 0;

            std::vector<ResultSet<size_t, float>> results(queryN);

            for (int qr = 0; qr < queryRepeat; ++qr) {
              for (size_t i = 0; i < queryN; ++i) {
                results[i] = ResultSet<size_t, float>(k);
              }

              std::cout << "progressiveIndex.knnSearch called()" << std::endl;
              timer.begin();
              progressiveIndex.knnSearch(queryPoints, results, k, searchParam);
              searchElapsed += timer.end();
              std::cout << "progressiveIndex.knnSearch done" << std::endl;
            }

            // check the result
            float meanDistError = 0;
            float accuracy = 0;

            measureMeanDistError(queryN, k, exactResults, results, meanDistError, accuracy);

            float qps = 1.0 / (searchElapsed / queryN / queryRepeat);
            log << "progressive" << addPointWeight << "\t";
            log << dataset.name << "\t" << dataset.version << "\t" << repeat << "\t" << maxOp << "\t" << r << "\t" << numPointsInserted << "\t";
            auto imbalances = progressiveIndex.recomputeImbalances();
            for (auto imbalance : imbalances) {
              log << imbalance << "\t";
            }
            log << progressiveIndex.computeMaxDepth() << "\t" << qps << "\t" << accuracy << "\t" << meanDistError << "\t";
            log << addNewPointElapsed << "\t" << updateIndexElapsed << std::endl;

            for (size_t i = 0; i < 4; ++i) {
              std::cout << "tree" << i << " " << progressiveIndex.trees[i]->getCachedImbalance() << " " << progressiveIndex.trees[i]->size << std::endl;
            }
            std::cout << "updateCost: " << progressiveIndex.getSize() * std::log2(progressiveIndex.getSize()) << std::endl;
            std::cout << "queryLoss: " << progressiveIndex.queryLoss << std::endl;
            if (progressiveIndex.updateStatus == progressiveIndex.BuildingTree) {
              std::cout << "building a tree backstage" << std::endl;
              progressiveIndex.printBackstage();
            }
            if (progressiveIndex.updateStatus == progressiveIndex.InsertingPoints) {
              progressiveIndex.printBackstage();
            }
          }          
        }
      }
    }
  }  

  log.close();
}

int main() {
  run();
  return 0;
}
