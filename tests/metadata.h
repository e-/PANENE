#ifndef panene_tests_metadata_h
#define panene_tests_metadata_h

#include <iostream>
#include <fstream>

#include <util/timer.h>
#include <util/matrix.h>
#include <binary_data_source.h>
#include <util/result_set.h>

// for test
#define BASE "D:\\G\\work\\panene\\PANENE\\data"

#define GLOVE_TRAIN_PATH(v) BASE "/glove/glove." #v ".bin"
#define GLOVE_QUERY_PATH BASE "/glove/test.bin"
#define GLOVE_ANSWER_PATH(v) BASE "/glove/glove." #v ".answer.txt"

#define SIFT_TRAIN_PATH(v) BASE "/sift/sift." #v ".bin"
#define SIFT_QUERY_PATH BASE "/sift/test.bin"
#define SIFT_ANSWER_PATH(v) BASE "/sift/sift." #v ".answer.txt"

namespace panene
{

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

void getExactNN(BinaryDataSource &dataset, Matrix<float> &queryset, Matrix<float> &answers, int k) {
  typedef float DistanceType;

  int rows = dataset.size();
  int n = k;
  int d = dataset.dim();

#pragma omp parallel for
  for (int q = 0; q < (int)queryset.rows; q++) {
    int* match = new int[n];
    DistanceType* dists = new DistanceType[n];

    float * query = queryset[q];

    dists[0] = 0;
    for (int j = 0; j < d; ++j) dists[0] += (dataset.get(0, j) - query[j]) * (dataset.get(0, j) - query[j]);
    match[0] = 0;
    int dcnt = 1;

    for (int i = 1; i<rows; ++i) {
      DistanceType tmp = 0;
      for (int j = 0; j < d; ++j) tmp += (dataset.get(i, j) - query[j]) * (dataset.get(i, j) - query[j]);
      
      if (dcnt<n) {
        match[dcnt] = i;
        dists[dcnt++] = tmp;
      }
      else if (tmp < dists[dcnt - 1]) {
        dists[dcnt - 1] = tmp;
        match[dcnt - 1] = i;
      }

      int j = dcnt - 1;
      // bubble up
      while (j >= 1 && dists[j]<dists[j - 1]) {
        std::swap(dists[j], dists[j - 1]);
        std::swap(match[j], match[j - 1]);
        j--;
      }
    }

    for (int i = 0; i<n; ++i) {
      answers[q][i] = sqrt(dists[i]);
    }

    delete[] match;
    delete[] dists;
  }
}

};

#endif
