#ifndef kdtree_test_h
#define kdtree_test_h

#include <flann/flann.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include "../indices/kd_tree_index.h"

namespace paknn {
using namespace std;

class KDTreeTest
{
  void generateRandomData(int rows, int cols, float *dataset) {
    for(int i = 0; i < rows * cols; ++i) {
      dataset[i] = (float)rand() / RAND_MAX;
    }
  }

  bool check(const flann::Matrix<size_t> &indices0, const paknn::Matrix<size_t> &indices1, int row, int k) {
    for(int i = 0; i < row; ++i) {
      for(int j = 0; j < k; ++j) {
        if(indices0[i][j] != indices1[i][j]) {
          cout << i << ", " << j << " expected " << indices0[i][j] << " but got " << indices1[i][j] << endl;
          return false;
        }
      } 
    }
    return true;
  }

public:
  void run() {
    int rows = 20000;
    int cols = 20;
    int k = 10;
    int q = 2000;
    srand(time(NULL));
    int seed = rand();

//    cout << "This test checks whether two versions of KDTrees (ours and FLANN's) works identically." << endl;

    float *data = new float[rows * cols];
    float *query = new float[q * cols];

    srand(0);
    generateRandomData(rows, cols, data);
    generateRandomData(q, cols, query);

    srand(seed);
    paknn::Matrix<float> paknnDataMatrix(data, rows, cols);
    paknn::Matrix<float> paknnQueryMatrix(query, q, cols);
    paknn::KDTreeIndex<paknn::L2<float>> paknnIndex(paknn::IndexParams(1));
    paknnIndex.setDataSource(paknnDataMatrix);
    paknnIndex.addPoints(rows);

    paknn::SearchParams paknnSearchParam(1);
    paknnSearchParam.cores = 1;

    paknn::Matrix<size_t> paknnIndices(new size_t[q * k], q, k);
    paknn::Matrix<float> paknnDists(new float[q * k], q, k);

    paknnIndex.knnSearch(paknnQueryMatrix, paknnIndices, paknnDists, k, paknnSearchParam);
   
    cerr << "Output of PAKNN's" << endl;
    for(int i = 0; i < q; ++i) {
      cerr << i << " : ";

      for(int j = 0; j < k; ++j) {
        cerr << "(" << paknnIndices[i][j] << ", " << paknnDists[i][j] << ") ";
      }

      cerr << endl;
    }

    srand(seed);
    flann::Matrix<float> flannDataMatrix(data, rows, cols);
    flann::Matrix<float> flannQueryMatrix(query, q, cols);
    flann::KDTreeIndex<flann::L2<float>> flannIndex(flannDataMatrix, flann::KDTreeIndexParams(1));

    flann::SearchParams flannSearchParam(1);
    flannSearchParam.cores = 1;

    flann::Matrix<size_t> flannIndices(new size_t[q * k], q, k);
    flann::Matrix<float> flannDists(new float[q * k], q, k);

    flannIndex.buildIndex();
    flannIndex.knnSearch(flannQueryMatrix, flannIndices, flannDists, k, flannSearchParam);
    
    cerr << "Output of FLANN's" << endl;
    for(int i = 0; i < q; ++i) {
      cerr << i << " : ";

      for(int j = 0; j < k; ++j) {
        cerr << "(" << flannIndices[i][j] << ", " << flannDists[i][j] << ") ";
      }

      cerr << endl;
    }

    if(check(flannIndices, paknnIndices, q, k)) {
      cout << "The results are the same." << endl;
    }
    else {
      cout << "The results are different." << endl;
    }
    
    delete[] data;
    delete[] query;
    delete[] flannIndices.ptr();
    delete[] flannDists.ptr();
  }
};

};

#endif

