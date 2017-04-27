#ifndef panene_kdtree_test_h
#define panene_kdtree_test_h

#include <flann/flann.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include "../indices/kd_tree_index.h"

namespace panene {
using namespace std;

class KDTreeTest
{
  void generateRandomData(int rows, int cols, float *dataset) {
    for(int i = 0; i < rows * cols; ++i) {
      dataset[i] = (float)rand() / RAND_MAX;
    }
  }

  bool check(const flann::Matrix<size_t> &indices0, const Matrix<size_t> &indices1, int row, int k) {
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

    cout << "This test checks whether two versions of KDTrees (ours and FLANN's) work identically." << endl;

    float *data = new float[rows * cols];
    float *query = new float[q * cols];

    srand(seed);
    generateRandomData(rows, cols, data);
    generateRandomData(q, cols, query);

    srand(seed);
    Matrix<float> paneneDataMatrix(data, rows, cols);
    Matrix<float> paneneQueryMatrix(query, q, cols);
    KDTreeIndex<L2<float>> paneneIndex(IndexParams(1));
    paneneIndex.setDataSource(paneneDataMatrix);
    paneneIndex.addPoints(rows);

    SearchParams paneneSearchParam(1);
    paneneSearchParam.cores = 1;

    Matrix<size_t> paneneIndices(new size_t[q * k], q, k);
    Matrix<float> paneneDists(new float[q * k], q, k);

    paneneIndex.knnSearch(paneneQueryMatrix, paneneIndices, paneneDists, k, paneneSearchParam);
   
    cerr << "Output of PAKNN's" << endl;
    for(int i = 0; i < q; ++i) {
      cerr << i << " : ";

      for(int j = 0; j < k; ++j) {
        cerr << "(" << paneneIndices[i][j] << ", " << paneneDists[i][j] << ") ";
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

    if(check(flannIndices, paneneIndices, q, k)) {
      cout << "The results are the same." << endl;
    }
    else {
      cout << "The results are different." << endl;
    }
    
    delete[] data;
    delete[] query;
    delete[] flannIndices.ptr();
    delete[] flannDists.ptr();
    delete[] paneneIndices.ptr();
    delete[] paneneDists.ptr();
  }
};

};

#endif

