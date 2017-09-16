#include <vector>
#include <map>

#include <tests/metadata.h>
#include <progressive_knn_table.h>
#include <kd_tree_index.h>

using namespace panene;

void run() {
  Timer timer;

  const size_t pointsN = 1000000;
  std::vector<Dataset> datasets = {    
    Dataset("sift", "shuffled",
      SIFT_TRAIN_PATH(shuffled),
      SIFT_QUERY_PATH,
      SIFT_ANSWER_PATH(shuffled),
      pointsN, 128),
    Dataset("sift", "original",
      SIFT_TRAIN_PATH(original),
      SIFT_QUERY_PATH,
      SIFT_ANSWER_PATH(original),
      pointsN, 128),
    Dataset("glove", "shuffled", 
        GLOVE_TRAIN_PATH(shuffled),
        GLOVE_QUERY_PATH,
        GLOVE_ANSWER_PATH(shuffled),
        pointsN, 100),
    Dataset("glove", "original", 
        GLOVE_TRAIN_PATH(original),
        GLOVE_QUERY_PATH,
        GLOVE_ANSWER_PATH(original),
        pointsN, 100)    
  };
  
  const int maxRepeat = 5; //5;
  const int queryRepeat = 1; // 5;
  const IndexParams indexParam(4);  
  const size_t maxIter = 2500; // if all data is read, it stops
  const size_t maxQueryN = 1000;

  SearchParams searchParam(1000); // 4096);
  searchParam.cores = 0;

  std::fstream log;

#ifdef _WIN32
  log.open(BASE "./log.tsv", std::fstream::out);
#else
  log.open("./log.tsv", std::fstream::out);
#endif

  log <<
    "method\tdata\tversion\trepeat\tmaxOp\titer\tnumPointsInserted\timbalance1\timbalance2\timbalance3\timbalance4\tmax_depth\tQPS\tAccuracy\tmeanDistError\taddNewPointElapsed\tupdateIndexElapsed" << std::endl;

  size_t maxOps[] = { 5000 };// , 10000 };
  size_t maxOpsN = sizeof(maxOps) / sizeof(size_t);

  float addPointWeights[] = { 0.1, 0.2, 0.3 }; // {0.25, 0.5, 0.75};
  size_t weightN = sizeof(addPointWeights) / sizeof(float);

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

            if (progressiveIndex.updateStatus != NoUpdate) {
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

            if (progressiveIndex.updateStatus != NoUpdate) {
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

int main() {
  run();
  return 0;
}
