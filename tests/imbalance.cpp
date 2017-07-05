#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#include <progressive_knn_table.h>
#include <kd_tree_index.h>
#include <binary_data_source.h>
#include <util/result_set.h>
#include <util/timer.h>
#include <util/matrix.h>

using namespace panene;

inline float distance(const BinaryDataSource &a, const size_t i,
                      const BinaryDataSource &b, const size_t j) {
  int d = a.dim();
  float sum = 0;
  for(int k = 0; k < d; ++k) {
    float x = a.get(i, k);
    float y = b.get(j, k);
    sum += (x - y) * (x - y);
  }
  return sum;
}

void getExactNN(const BinaryDataSource &train, 
                const BinaryDataSource &test, 
                std::vector<ResultSet<size_t, float>> &exactResults, 
                size_t k) {

  size_t trainN = train.size();
  size_t d = train.dim();
  size_t testN = test.size();

#pragma omp parallel for
  for(size_t q = 0; q < testN; q++) {
    auto &result = exactResults[q];

    for(size_t i = 0; i < trainN; i++) {
      float dist = distance(train, i, test, q);
      if(!result.full() || result.worstDist > dist) {
        result << Neighbor<size_t, float>(i, dist);
      }
    }
  }
}

struct Dataset {
  std::string name;
  std::string version;
  std::string path;
  size_t n;
  size_t dim;

  Dataset() = default;
  Dataset(std::string name_, std::string version_, std::string path_, size_t n_, size_t dim_) : name(name_), version(version_), path(path_), n(n_), dim(dim_) {  }
};

void run() {
  Timer timer;

  std::vector<Dataset> datasets = {
    Dataset("glove", "original", "../../data/glove/glove.original.bin", 1100000, 100),
    Dataset("glove", "shuffled", "../../data/glove/glove.shuffled.bin", 1100000, 100),
    Dataset("glove", "sorted", "../../data/glove/glove.sorted.bin", 1100000, 100),
    Dataset("sift", "original", "../../data/sift/sift.original.bin", 1000000, 128),
    Dataset("sift", "shuffled", "../../data/sift/sift.shuffled.bin", 1000000, 128),
    Dataset("sift", "sorted", "../../data/sift/sift.sorted.bin", 1000000, 128),
    Dataset("gist", "original", "../../data/gist/gist.original.bin", 1000000, 960),
    Dataset("gist", "shuffled", "../../data/gist/gist.shuffled.bin", 1000000, 960),
    Dataset("gist", "sorted", "../../data/gist/gist.sorted.bin", 1000000, 960)
  };
  
  const int k = 20;
  const int maxOps = 1024;
  const int maxRepeat = 10; //0; //5;
  const int maxIter = 1000;
  
  SearchParams searchParam(10000);
  searchParam.cores = 0;

/*  std::cout << "computing exact neighbors for test points (N = " << testN << ")" << std::endl;
  
  std::vector<ResultSet<size_t, float>> exactResults(testN);
  std::vector<std::vector<float>> testPoints(testN);
  
  for(size_t i = 0; i < testN; ++i) {
    exactResults[i] = ResultSet<size_t, float>(k);
    std::vector<float> point(d);

    for(size_t j = 0; j < d; ++j) point[j] = testDataSource.get(i, j);
    testPoints[i] = point;
  }

  getExactNN(trainDataSource, testDataSource, exactResults, k);

  std::cout << "exact neighbors computation done" << std::endl;*/

 
  std::cerr <<
    "data\tversion\trepeat\titer\tdata_inserted\timbalance1\timbalance2\timbalance3\timbalance4\tmax_depth" << std::endl;
  //std::cerr << "method\trepeat\titeration\taddNewPointRes\taddNewPointElpased\tupdateIndexRes\tupdateIndexElapsed\tsearchElpased\taccuracy\tmeanDistanceError" << std::endl;

  float addPointWeights[] = {1}; //.4, .5, .6, .7};
  size_t weightN = sizeof(addPointWeights) / sizeof(float);


  for(const auto& dataset: datasets) {
    panene::BinaryDataSource trainDataSource(dataset.path);

    size_t trainN = trainDataSource.open(dataset.path, dataset.n, dataset.dim);

    for(int w = 0; w < weightN; ++w) {
      float weight = addPointWeights[w];
      if(maxOps * weight * maxIter < trainN ) {
        std::cout << "weight " << weight << " is too small. Some points may not be indexed" << std::endl;
      }
    } 

    for(int w = 0; w < weightN; ++w) {
      float addPointWeight = addPointWeights[w];

      for(int repeat = 0; repeat < maxRepeat; ++repeat) {
        ProgressiveKDTreeIndex<L2<float>, BinaryDataSource> progressiveIndex(IndexParams(4));
        progressiveIndex.setDataSource(&trainDataSource);
       
        for(int r = 0; r < maxIter; ++r) { 
          std::cout << "(" << w << "/" << weightN << ") (" << repeat << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;
          // update the index with the given number operations

          timer.begin();
          size_t addNewPointResult = progressiveIndex.addPoints(maxOps * addPointWeight);
          double addNewPointElapsed = timer.end();
          if(addNewPointResult == 0) break;
          
          timer.begin();
          size_t updateIndexResult = progressiveIndex.update(maxOps - addNewPointResult);
          double updateIndexElapsed = timer.end();

          auto imbalances = progressiveIndex.recomputeImbalances();
          
          std::cerr << dataset.name << "\t" << dataset.version << "\t" << repeat << "\t" << r << "\t" << addNewPointResult << "\t";
          for(auto imbalance : imbalances) {
            std::cerr << imbalance << "\t";
          } 
          std::cerr << progressiveIndex.computeMaxDepth() << std::endl;

          continue;

  /*        std::cout << "maintain: " << std::endl;
          auto imbalances2 = progressiveIndex.getImbalances();

          for(auto imbalance : imbalances) {
            std::cout << imbalance << std::endl;
          }*/
          

          // calculate accurarcy and mean distance error

  /*        std::vector<ResultSet<size_t, float>> results(testN);
          for(size_t i = 0; i < testN; ++i) {
            results[i] = ResultSet<size_t, float>(k); 
          }                      
         
          timer.begin(); 
          progressiveIndex.knnSearch(testPoints, results, k, searchParam); 
          double searchElapsed = timer.end();

          // check the result
          float meanDistError = 0;
          float accuracy = 0;

          for(size_t i = 0; i < testN; ++i) {
            meanDistError += results[i][k - 1].dist / exactResults[i][k - 1].dist;

            for(size_t j = 0; j < k; ++j) {
              for(size_t l = 0; l < k; ++l) {
                if(results[i][j] == exactResults[i][l]) accuracy += 1;
              }
            }
          }

          meanDistError /= testN;
          accuracy /= testN * k;

          std::cerr << "progressive" << addPointWeight << "\t" << repeat << "\t" << r << "\t" 
            << addNewPointResult << "\t" << addNewPointElapsed << "\t" 
            << updateIndexResult << "\t" << updateIndexElapsed << "\t"
            << searchElapsed << "\t" << accuracy << "\t" << meanDistError << std::endl;*/
        }
        auto logs = progressiveIndex.getInsertionLogs();
        std::map<int, int> dict;

        for(auto& log: logs) {
          for(auto& leaf: log){ 
            if(dict.count(leaf.count) == 0)
              dict[leaf.count] = 0;
            dict[leaf.count]++;
          }
        }

        for (auto& iter : dict) {
          std::cout << iter.first << " : " << iter.second << std::endl;
        }
      }
    }
  }
  
/*  for(int w = 0; w < weightN; ++w) {
    float addPointWeight = addPointWeights[w];

    for(int repeat = 0; repeat < maxRepeat; ++repeat) {
      KDTreeIndex<L2<float>, BinaryDataSource> index(IndexParams(4));
      index.setDataSource(&trainDataSource);
     
      for(int r = 0; r < trainN / (maxOps * addPointWeight) + 1; ++r) { 
        std::cout << "(" << w << "/" << weightN << ") (" << repeat << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;
        // update the index with the given number operations

        timer.begin();
        size_t addNewPointResult = index.addPoints(maxOps * addPointWeight * (r + 1));
        double addNewPointElapsed = timer.end();
        
        size_t updateIndexResult = 0;
        double updateIndexElapsed = 0;

        // calculate accurarcy and mean distance error

        std::vector<ResultSet<size_t, float>> results(testN);
        for(size_t i = 0; i < testN; ++i) {
          results[i] = ResultSet<size_t, float>(k); 
        }                      
       
        timer.begin(); 
        index.knnSearch(testPoints, results, k, searchParam); 
        double searchElapsed = timer.end();

        // check the result
        float meanDistError = 0;
        float accuracy = 0;

        for(size_t i = 0; i < testN; ++i) {
          meanDistError += results[i][k - 1].dist / exactResults[i][k - 1].dist;

          for(size_t j = 0; j < k; ++j) {
            for(size_t l = 0; l < k; ++l) {
              if(results[i][j] == exactResults[i][l]) accuracy += 1;
            }
          }
        }

        meanDistError /= testN;
        accuracy /= testN * k;

        std::cerr << "kd" << addPointWeight << "\t" << repeat << "\t" << r << "\t" 
          << addNewPointResult << "\t" << addNewPointElapsed << "\t" 
          << updateIndexResult << "\t" << updateIndexElapsed << "\t"
          << searchElapsed << "\t" << accuracy << "\t" << meanDistError << std::endl;
      }
    }
  }*/

}

int main() {
  run();
  return 0;
}
