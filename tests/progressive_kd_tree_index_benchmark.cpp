#include <iostream>
#include <vector>
#include <algorithm>

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
  for(int q = 0; q < (int)testN; q++) {
    auto &result = exactResults[q];

    for(size_t i = 0; i < trainN; i++) {
      float dist = distance(train, i, test, q);
      if(!result.full() || result.worstDist > dist) {
        result << Neighbor<size_t, float>(i, dist);
      }
    }
  }
}

void run() {
  panene::BinaryDataSource trainDataSource;
  panene::BinaryDataSource testDataSource;
  
  trainDataSource.open("../../data/glove/train.bin", 400000, 100);
  testDataSource.open("../../data/glove/test.bin", 100, 100);

  size_t trainN = trainDataSource.size();
  size_t testN = testDataSource.size();
  size_t d = trainDataSource.dim();
  Timer timer;
  const int k = 20;
  const int maxOps = 8092;
  const int maxRepeat = 10; //5;
  const int maxIter = 300;
  
  SearchParams searchParam(10000);
  searchParam.cores = 0;

  std::cout << "computing exact neighbors for test points (N = " << testN << ")" << std::endl;
  
  std::vector<ResultSet<size_t, float>> exactResults(testN);
  std::vector<std::vector<float>> testPoints(testN);
  
  for(size_t i = 0; i < testN; ++i) {
    exactResults[i] = ResultSet<size_t, float>(k);
    std::vector<float> point(d);

    for(size_t j = 0; j < d; ++j) point[j] = testDataSource.get(i, j);
    testPoints[i] = point;
  }

  getExactNN(trainDataSource, testDataSource, exactResults, k);

  std::cout << "exact neighbors computation done" << std::endl;

 
  std::cerr << "method\trepeat\titeration\taddNewPointRes\taddNewPointElpased\tupdateIndexRes\tupdateIndexElapsed\tsearchElpased\taccuracy\tmeanDistanceError" << std::endl;

  float addPointWeights[] = {.4, .5, .6, .7};
  size_t weightN = sizeof(addPointWeights) / sizeof(float);
 
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
        
        timer.begin();
        size_t updateIndexResult = progressiveIndex.update(maxOps - addNewPointResult);
        double updateIndexElapsed = timer.end();

        // calculate accurarcy and mean distance error

        std::vector<ResultSet<size_t, float>> results(testN);
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
          << searchElapsed << "\t" << accuracy << "\t" << meanDistError << std::endl;
      }
    }
  }

  for(int w = 0; w < weightN; ++w) {
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
  }

}

int main() {
  run();
  return 0;
}
