#include <string>
#include <fstream>
#include <algorithm>
#include <vector>

#include <progressive_knn_table.h>
#include <naive_data_source.h>
#include <util/timer.h>
#include <util/matrix.h>

class ProgressiveKNNTableBenchmark {
  NaiveDataSource* dataSource;

public:
  ProgressiveKNNTableBenchmark(NaiveDataSource* dataSource_) : dataSource(dataSource_) {}

  std::string getName() { return "ProgressiveKNNTableBenchmark"; }

  void getExactNN(Matrix<float> &dataset, Matrix<float> &queryset, Matrix<float> &answers, int k) {
    L2<float> distance = L2<float>();
    typedef typename L2<float>::ResultType DistanceType;

    int rows = dataset.rows;
    int n = k;

#pragma omp parallel for
    for (unsigned int q = 0; q < queryset.rows; q++) {
      int* match = new int[n];
      DistanceType* dists = new DistanceType[n];

      float * query = queryset[q];

      dists[0] = distance(dataset[0], query, dataset.cols);
      match[0] = 0;
      int dcnt = 1;

      for (int i = 1; i<rows; ++i) {
        DistanceType tmp = distance(dataset[i], query, dataset.cols);

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
        answers[q][i] = dists[i];
      }

      delete[] match;
      delete[] dists;
    }
  }

  void run() {
    size_t n = dataSource->size();
    size_t d = dataSource->dim();
    int k = 10;
    int maxOps = 3000;
    Timer timer;

    //srand(time(0));

    std::cerr << getName() << " starts with n = " << n << ", d = " << d << ", k = " << k << ", maxOps = " << maxOps << std::endl;

    float *numbers = new float[n * d];

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < d; ++j) {
        numbers[i * d + j] = dataSource->get(i, j);
      }
    }

    SearchParams searchParam(512);
    searchParam.cores = 1;

    // create a progressive table
    ProgressiveKNNTable<ProgressiveKDTreeIndex<L2<float>, NaiveDataSource>, NaiveDataSource> table(k + 1, d, IndexParams(4), searchParam);
    std::cerr << "progressive knn table created" << std::endl;

    Matrix<float> dataMatrix(numbers, n, d);
    table.setDataSource(dataSource);

    // since the raw data are can be large, we create a sample of points that will be tested.
    // sample 5% of points

    size_t sample = n / 20;
    std::vector<size_t> order(n);

    for (size_t i = 0; i < n; ++i) order[i] = i;
    random_shuffle(order.begin(), order.end());

    Matrix<float> samplePoints(new float[sample * d], sample, d);

    for (size_t i = 0; i < sample; ++i)
      for (size_t j = 0; j < d; ++j)
        samplePoints[i][j] = dataMatrix[order[i]][j];

    std::cerr << "sampled " << sample << " points" << std::endl;

    // compute the exact knn of the sample
    Matrix<float> exactDists(new float[sample * (k + 1)], sample, k + 1);

    getExactNN(dataMatrix, samplePoints, exactDists, k + 1);

    std::cerr << "computed the exact nn of the sample" << std::endl;

    std::cerr << "iteration\taddNewPointRes\tupdateIndexRes\tupdateTableRes\tsampleSize\tcoverage\tmeanDistanceError\ttime" << std::endl;

    for (int r = 0; r < 500; ++r) {
      // update the table with the given number operations
      timer.begin();
      auto result = table.update(maxOps);
      double elapsed = timer.end();

      // calculate the mean distance error
      // since our table is progressive, some points in the sample can be missing in the table.
      // therefore, we need to count the number of such cases

      auto size = table.getSize();
      size_t missing = 0;
      float distSum = 0;

      for (size_t i = 0; i < sample; ++i) {
        size_t id = order[i];

        if (id < size) { // the sample point is present in the table
          auto nn = table.getNeighbors(id);

          // get the distance to the farthest neighbor
          float dist = nn[k].dist;

          //          std::cerr << nn << std::endl;

          // get the exact distnce
          float exact = exactDists[i][k];

          //          std::cerr << dist << " " << exact << std::endl;
          distSum += dist / exact;
        }
        else {
          missing++;
        }
      }

      if (missing == sample) {
        std::cerr << "all points are missing" << std::endl;
        continue;
      }

      float meanError = distSum / (sample - missing);

      std::cerr << r << "\t" << result.addNewPointResult << "\t"
        << result.updateIndexResult << "\t"
        << result.updateTableResult << "\t"
        << sample << "\t" << sample - missing << "\t" << meanError << "\t" << elapsed << std::endl;

    }

    delete[] samplePoints.ptr();
    delete[] exactDists.ptr();
  }
};


int main(){
  panene::NaiveDataSource naiveDataSource;
  
  naiveDataSource.open("../../data/sift.shuffled.txt", 100000, 128);

  ProgressiveKNNTableBenchmark test(&naiveDataSource);

  test.run();
  return 0;
}
