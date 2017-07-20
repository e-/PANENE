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

void readAnswers(const std::string &path, size_t queryN, size_t &k, std::vector<std::vector<Neighbor<size_t, float>>> &neighbors){
  FILE *fd;
  fd = fopen(path.c_str(), "rb"); 
  if(!fd) {
    std::cerr << "file " << path << " does not exist" << std::endl;
    throw;
  }

  auto ret = fread(&k, sizeof(size_t), 1, fd);

  neighbors.resize(queryN);
  for(size_t i = 0; i < queryN; ++i) {
    neighbors[i].resize(k);

    for(size_t j = 0 ; j < k; ++j) {
      auto ret = fread(&neighbors[i][j].id, sizeof(size_t), 1, fd);
      ret = fread(&neighbors[i][j].dist, sizeof(float), 1, fd);
    }
  }

  fclose(fd);
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

#define GLOVE_TRAIN_PATH(v) "../../data/glove/glove." #v ".bin"
#define GLOVE_QUERY_PATH "../../data/glove/test.bin"
#define GLOVE_ANSWER_PATH(v) "../../data/glove/glove." #v ".answer.bin"

#define SIFT_TRAIN_PATH(v) "../../data/sift/sift." #v ".bin"
#define SIFT_QUERY_PATH "../../data/sift/test.bin"
#define SIFT_ANSWER_PATH(v) "../../data/sift/sift." #v ".answer.bin"

void run() {
  Timer timer;

  std::vector<Dataset> datasets = {
/*    Dataset("glove", "shuffled", 
        GLOVE_TRAIN_PATH(shuffled),
        GLOVE_QUERY_PATH,
        GLOVE_ANSWER_PATH(shuffled),
        1100000, 100),
    Dataset("glove", "original", 
        GLOVE_TRAIN_PATH(original),
        GLOVE_QUERY_PATH,
        GLOVE_ANSWER_PATH(original),
        1100000, 100),*/
    Dataset("sift", "shuffled", 
        SIFT_TRAIN_PATH(shuffled),
        SIFT_QUERY_PATH,
        SIFT_ANSWER_PATH(shuffled),
        1000000, 128),
    Dataset("sift", "original", 
        SIFT_TRAIN_PATH(original),
        SIFT_QUERY_PATH,
        SIFT_ANSWER_PATH(original),
        1000000, 128)
/*    Dataset("glove", "shuffled", "../../data/glove/glove.shuffled.bin", 1100000, 100),
    Dataset("glove", "sorted", "../../data/glove/glove.sorted.bin", 1100000, 100),
    Dataset("sift", "original", "../../data/sift/sift.original.bin", 1000000, 128),
    Dataset("sift", "shuffled", "../../data/sift/sift.shuffled.bin", 1000000, 128),
    Dataset("sift", "sorted", "../../data/sift/sift.sorted.bin", 1000000, 128),
    Dataset("gist", "original", "../../data/gist/gist.original.bin", 1000000, 960),
    Dataset("gist", "shuffled", "../../data/gist/gist.shuffled.bin", 1000000, 960),
    Dataset("gist", "sorted", "../../data/gist/gist.sorted.bin", 1000000, 960)*/
  };
  
  const int maxOps = 1024;
  const int maxRepeat = 1; //0; //5;
  const int maxIter = 500;
  const int queryRepeat = 5;
  
  SearchParams searchParam(5000);
  searchParam.cores = 0;

  std::cerr <<
    "method\tdata\tversion\trepeat\titer\tnumPointsInserted\timbalance1\timbalance2\timbalance3\timbalance4\tmax_depth\tQPS\tAccuracy\tmeanDistError" << std::endl;

  float addPointWeights[] = {1, .5}; //.4, .5, .6, .7};
  size_t weightN = sizeof(addPointWeights) / sizeof(float);

  for(const auto& dataset: datasets) {
    panene::BinaryDataSource trainDataSource(dataset.path);

    size_t trainN = trainDataSource.open(dataset.path, dataset.n, dataset.dim);
    
    // read query set
    panene::BinaryDataSource queryDataSource(dataset.queryPath);
    
    size_t queryN = queryDataSource.open(dataset.queryPath, 10000, dataset.dim);

    std::vector<std::vector<float>> queryPoints(queryN);

    for(size_t i = 0; i < queryN; ++i) {
      std::vector<float> point(dataset.dim);

      for(size_t j = 0; j < dataset.dim; ++j) point[j] = queryDataSource.get(i, j);
      queryPoints[i] = point;
    }

    for(int w = 0; w < weightN; ++w) {
      float weight = addPointWeights[w];
      if(maxOps * weight * maxIter < trainN ) {
        std::cout << "weight " << weight << " is too small. Some points may not be indexed" << std::endl;
      }
    } 

    // read k and answers

    size_t k;
    std::vector<std::vector<Neighbor<size_t, float>>> exactResults;
    
    readAnswers(dataset.answerPath, queryN, k, exactResults);

    for(int w = 0; w < weightN; ++w) {
      float addPointWeight = addPointWeights[w];
      size_t numPointsInserted = 0;

      for(int repeat = 0; repeat < maxRepeat; ++repeat) {
        ProgressiveKDTreeIndex<L2<float>, BinaryDataSource> progressiveIndex(IndexParams(4));
        progressiveIndex.setDataSource(&trainDataSource);
        
        for(int r = 0; r < maxIter; ++r) { 
          std::cout << "(" << w << "/" << weightN << ") (" << repeat << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;
          // update the index with the given number operations

          size_t addPointOps;

          if(addPointWeight < 1) { // see imbalances
            auto imbalances = progressiveIndex.recomputeImbalances();
            bool violation = false;
            for(auto imbalance : imbalances) {
              if(imbalance > 1.2) violation = true;
            }

            if(violation) { 
              addPointOps = maxOps * addPointWeight;
            }
            else {
              addPointOps = maxOps;
            }
          }
          else { // put all operations to addition
            addPointOps = maxOps;
          }

          timer.begin();
          size_t addNewPointResult = progressiveIndex.addPoints(addPointOps);
          double addNewPointElapsed = timer.end();
          if(addNewPointResult == 0) break;
          numPointsInserted += addNewPointResult;
          
          timer.begin();
          size_t updateIndexResult = progressiveIndex.update(maxOps - addNewPointResult);
          double updateIndexElapsed = timer.end();

          auto imbalances = progressiveIndex.recomputeImbalances();
          
  /*        for(auto imbalance : imbalances) {
            std::cout << imbalance << "\t";
          } 
          std::cout << std::endl;

          auto imbalances2 = progressiveIndex.getCachedImbalances();

          for(auto imbalance : imbalances2) {
            std::cout << imbalance << "\t";
          } 
          std::cout << std::endl;
*/

  /*        std::cout << "maintain: " << std::endl;
          auto imbalances2 = progressiveIndex.getImbalances();

          for(auto imbalance : imbalances) {
            std::cout << imbalance << std::endl;
          }*/
          

          // calculate accurarcy and mean distance error

          double searchElapsed = 0;
          
          std::vector<ResultSet<size_t, float>> results(queryN);

          for(int qr = 0; qr < queryRepeat; ++qr) {
            for(size_t i = 0; i < queryN; ++i) {
              results[i] = ResultSet<size_t, float>(k); 
            }                      
        
            timer.begin(); 
            progressiveIndex.knnSearch(queryPoints, results, k, searchParam); 
            searchElapsed += timer.end();
          }

          // check the result
          float meanDistError = 0;
          float accuracy = 0;

          for(size_t i = 0; i < queryN; ++i) {
            meanDistError += results[i][k - 1].dist / exactResults[i][k - 1].dist;

            for(size_t j = 0; j < k; ++j) {
              for(size_t l = 0; l < k; ++l) {
                if(results[i][j] == exactResults[i][l]) accuracy += 1;
              }
            }
          }

          meanDistError /= queryN;
          accuracy /= queryN * k;
          float qps = 1 / (searchElapsed / queryN / queryRepeat);
          std::cerr << "progressive" << addPointWeight << "\t";
          std::cerr << dataset.name << "\t" << dataset.version << "\t" << repeat << "\t" << r << "\t" << numPointsInserted << "\t";
          for(auto imbalance : imbalances) {
            std::cerr << imbalance << "\t";
          } 
          std::cerr << progressiveIndex.computeMaxDepth() << "\t" << qps << "\t" << accuracy << "\t" << meanDistError << std::endl;
//          std::cerr << addNewPointElapsed << " " << updateIndexElapsed << std::endl; 
        }
        const auto dict = progressiveIndex.computeCountDistribution();

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
