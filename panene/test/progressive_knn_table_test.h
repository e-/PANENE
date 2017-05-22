#ifndef panene_progressive_knn_table_test_h
#define panene_progressive_knn_table_test_h

#include <flann/flann.hpp>
#include <string>
#include <fstream>

#include "test.h"
#include "../progressive_knn_table.h"
#include "../indices/progressive_kd_tree_index.h"

using namespace std;

namespace panene {

class ProgressiveKNNTableTest : Test {
public:
  ProgressiveKNNTableTest() = default;

  std::string getName() { return "ProgressiveKNNTableTest"; }

  void readData(const std::string &path, int rows, int cols, float *dataset) {
    std::ifstream ifs(path);

    for(int i = 0; i < rows * cols; ++i) {
      ifs >> dataset[i]; 
    }
  }

  void randomData(int rows, int cols, float *dataset) {
    for(int i = 0; i < rows * cols; ++i) {
      dataset[i] = (float)rand() / RAND_MAX;
    }
  }

  void getCorrectNN(flann::Matrix<float> &dataset, flann::Matrix<float> &queryset, flann::Matrix<float> &answers, int k){
    flann::L2<float> distance = flann::L2<float>();
      typedef typename flann::L2<float>::ResultType DistanceType;
      
      int rows = dataset.rows;
      int n = k;

#pragma omp parallel for
      for(unsigned int q = 0; q < queryset.rows; q++) {
          int* match = new int[n];
          DistanceType* dists = new DistanceType[n];

          float * query = queryset[q];

          dists[0] = distance(dataset[0], query, dataset.cols);
          match[0] = 0;
          int dcnt = 1;

          for (int i=1; i<rows; ++i) {
              DistanceType tmp = distance(dataset[i], query, dataset.cols);

              if (dcnt<n) {
                  match[dcnt] = i;
                  dists[dcnt++] = tmp;
              }
              else if (tmp < dists[dcnt-1]) {
                  dists[dcnt-1] = tmp;
                  match[dcnt-1] = i;
              }

              int j = dcnt-1;
              // bubble up
              while (j>=1 && dists[j]<dists[j-1]) {
                  std::swap(dists[j],dists[j-1]);
                  std::swap(match[j],match[j-1]);
                  j--;
              }
          }

          for (int i=0; i<n; ++i) {
              answers[q][i] = dists[i];
          }

          delete[] match;
          delete[] dists;
      }
  }

  //#define DATA_PATH "../data/glove.shuffled.txt"
  //#define D 100

  //#define DATA_PATH "../data/sift.shuffled.txt"
  //#define D 128

#define DATA_PATH "random"
#define D 100

  //#define DATA_PATH "../data/creditcard.shuffled.txt"
  //#define D 28

  void run(){
    int n = 10000; // # of rows
    int d = D;
    int k = 10;
    int sample = 1000;
    
    //srand(time(0));

    float *data = new float[n * d];
    string path = DATA_PATH;

    cerr << "test start" << endl;
    cerr << path << endl;

    if(path.compare("random") == 0) {
      randomData(n, d, data);
      cerr << "data:" << data[0] << " " << data[1] << endl;
    }
    else {
      readData(DATA_PATH, n, d, data);
    }

    cerr << "data read" << endl;

    SearchParams searchParam(512);
    searchParam.cores = 1;

    flann::SearchParams flannSearchParam(512);
    flannSearchParam.cores = 1;

    ProgressiveKNNTable<ProgressiveKDTreeIndex<L2<float>>> table(k + 1, d, IndexParams(4), searchParam);
    cerr << "progressive knn table created" << endl;
    Matrix<float> allData(data, n, d);
    table.setDataSource(allData);

    flann::Matrix<float> initData(nullptr, 0, d);
    flann::Index<flann::L2<float>> index(initData, flann::KDTreeIndexParams(4)); // baseline 

    cerr << "baseline created" << endl;

    flann::Matrix<size_t> indices(new size_t[sample * (k + 1)], sample, k + 1);
    flann::Matrix<float> dists(new float[sample * (k + 1)], sample, k + 1);

    cout << "benchmark for " << DATA_PATH << " with " << D << " dimensions, " << n << " rows." << endl;
    cout << "updated\trows\taccuracy_table\taccuracy_index" << endl;


    for(int r = 0; r < 3; ++r) { 
      cerr << "update(" << r << ")" << endl;
      auto ar = table.update(100);

      cerr << ar << endl;

      //flann::Matrix<float> chunkMatrix(data + r * n * d, n, d);
      //vector<size_t> ids;
      //flann::Matrix<float> dataMatrix(data, n * (r + 1), d);
      //flann::Matrix<float> queryMatrix(new float[sample * d], sample, d);
      //flann::Matrix<float> nns(new float[sample * (k + 1)], sample, k + 1);
   
      //// We cannot check all rows to calculate accuracy due to the O(n^2) 
      //// complexity of the exact algorithm. 
      //// Instead, we make a small sample and approximate the accuracy.

      //for(int s = 0; s < sample; ++s) {
        //size_t id = rand() % (n * (r + 1));

        //ids.push_back(id);
        //for(int i = 0; i < d; ++i) {
          //queryMatrix[s][i] = dataMatrix[id][i];
        //} 
      //}

      //// get exact NNs for the sample
      //getCorrectNN(dataMatrix, queryMatrix, nns, k + 1);
      //cerr << "got correct NN" << endl;

      //// for comparison, measure the performance of KDtrees
      //index.addPoints(chunkMatrix);
      //cerr << "added point to the baseline index" << endl;

      //// compare the kNN with the ones in the KNN table
      //int correct_table = 0, correct_tree = 0;

      //for(int s = 0; s < sample; ++s) {
        //size_t id = ids[s];
        //float radius = nns[s][k];
        //int kk = k;

        //for(int i = 0; i < kk; ++i) {
          //auto &ne = table.getNeighbors(id)[i];

          //if(id != ne.id && ne.distance <= radius)
            //correct_table++;
          //if(id == ne.id)
            //kk = k + 1;
        //}
      //}
      
      //// do kNN search using KDTree
      //index.knnSearch(queryMatrix, indices, dists, k + 1, flannSearchParam);

      //for(int s = 0; s < sample; ++s) {
        //size_t id = ids[s];
        //float radius = nns[s][k];
        //int kk = k;

        //for(int i = 0; i < kk; ++i) {
          //if(id != indices[s][i] && dists[s][i] <= radius)
            //correct_tree++;
          //if(id == indices[s][i])
            //kk = k + 1;
        //}
      //}
      
      //float accuracy_table = (float)correct_table / sample / k;
      //float accuracy_tree = (float)correct_tree / sample / k;

      //cout << (r + 1) * n << '\t' << accuracy_table << '\t' << accuracy_tree << endl;
    
      //delete[] queryMatrix.ptr(); 
      //delete[] nns.ptr();
    }

    delete[] data;
    delete[] indices.ptr();
    delete[] dists.ptr();
  }
};

}

#endif
