#include <string>
#include <vector>

#define BENCHMARK

#include <tests/metadata.h>
#include <progressive_knn_table.h>

using namespace panene;

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

  const int maxRepeat = 1; //5;
  const int queryRepeat = 10000;
  const IndexParams indexParam(4);
  const size_t maxIter = 1500; // if all data is read, it stops
  const size_t maxQueryN = 1000;

  SearchParams searchParam(1024); // 4096);
  searchParam.cores = 0;

  std::fstream log;

#ifdef _WIN32
  log.open(BASE "./knn_table_log.tsv", std::fstream::out);
#else
  log.open("./knn_table_log.tsv", std::fstream::out);
#endif

  size_t maxOps[] = { 5000 };// , 10000 };
  size_t maxOpsN = sizeof(maxOps) / sizeof(size_t);

  float addPointWeights[] = { 0.3, 0.5};
  size_t weightN = sizeof(addPointWeights) / sizeof(float);
  size_t k = 20;

  log << "data\tversion\taddPointWeight\tmaxOp\tnumPointsInserted\t"
    << "iteration\taddPointsRes\tupdateIndexRes\tupdateTableRes\t"
    << "addPointsElapsed\tupdateIndexElapsed\tupdateTableElapsed\t" 
    << "meanDistError\tupdateElapsed\tqueryElapsed\tQPS\tcoverage" << std::endl;

  /*
  std::cerr << dataset.name << "\t" << dataset.version << "\t" << addPointWeight << "\t" << maxOp << "\t" << numPointsInserted << "\t"
              << r << "\t" << updateResult.addPointsResult << "\t" << updateResult.updateIndexResult << "\t" << updateResult.updateTableResult << "\t"
              << updateResult.addPointsElapsed << "\t" << updateResult.updateIndexElapsed << "\t" << updateResult.updateTableElapsed << "\t"
              << meanError << "\t" << updateElapsed << "\t" << queryElapsed << "\t" << qps << "\t" << coverage << std::endl;
    */

  for (const auto& dataset : datasets) {
    panene::BinaryDataSource trainDataSource(dataset.path);

    size_t trainN = trainDataSource.open(dataset.path, dataset.n, dataset.dim);

    // create query set

    size_t sample = trainN / 1000;
    std::vector<size_t> order(trainN);

    for (size_t i = 0; i < trainN; ++i) order[i] = i;
    random_shuffle(order.begin(), order.end());

    Matrix<float> samplePoints(new float[sample * trainDataSource.dim()], sample, trainDataSource.dim());

    for (size_t i = 0; i < sample; ++i)
      for (size_t j = 0; j < trainDataSource.dim(); ++j)
        samplePoints[i][j] = trainDataSource.get(order[i], j);

    std::cerr << "sampled " << sample << " points" << std::endl;

    // compute the exact knn of the sample
    Matrix<float> exactDists(new float[sample * k], sample, k);
    getExactNN(trainDataSource, samplePoints, exactDists, k);

    std::cerr << "computed the exact nn of the sample" << std::endl;

    for (int repeat = 0; repeat < maxRepeat; ++repeat) {
      for (const size_t maxOp : maxOps) {
        for (int w = 0; w < weightN; ++w) {
          float addPointWeight = addPointWeights[w];
          size_t numPointsInserted = 0;

          ProgressiveKNNTable<ProgressiveKDTreeIndex<L2<float>, BinaryDataSource>, BinaryDataSource> table(k, dataset.dim, IndexParams(4), searchParam,
            WeightSet(addPointWeight, 0.3));

          table.setDataSource(&trainDataSource);

          for (int r = 0; r < maxIter; ++r) {
            std::cout << "(" << w << "/" << weightN << ") (" << repeat << "/" << maxRepeat << ") (" << r << "/" << maxIter << ")" << std::endl;
            // update the index with the given number operations

            std::cout << "table.update called() " << std::endl;
            timer.begin();
            UpdateResult updateResult = table.update(maxOp);
            double updateElapsed = timer.end();
            std::cout << "table.update done" << std::endl;

            if (updateResult.addPointsResult == 0) break;            

            std::vector<ResultSet<size_t, float>> results(sample);
            size_t missing = 0;
            float distSum = 0;

            timer.begin();
            for (int qr = 0; qr < queryRepeat; ++qr) {
              for (size_t i = 0; i < sample; ++i) {
                size_t id = order[i];

                if (id < updateResult.numPointsInserted) { // the sample point is present in the table
                  auto nn = table.getNeighbors(id);

                  // get the distance to the farthest neighbor
                  float dist = nn[k - 1].dist;

                  // get the exact distnce
                  float exact = exactDists[i][k - 1];

                  //std::cerr << dist << " " << exact << std::endl;
                  distSum += dist / exact;
                }
                else {
                  missing++;
                }
              }
            }
            double queryElapsed = timer.end();
            
            float qps = 1.0 / (queryElapsed / sample / queryRepeat);

            if (missing == sample) {
              std::cerr << "all points are missing" << std::endl;
              continue;
            }

            //std::cerr << distSum << " " << missing << " " << queryRepeat << std::endl;
            float meanError = distSum / (queryRepeat * sample - missing);
            float coverage = 1 - (float)missing / queryRepeat / sample;

            log << dataset.name << "\t" << dataset.version << "\t" << addPointWeight << "\t" << maxOp << "\t" << numPointsInserted << "\t"
              << r << "\t" << updateResult.addPointsResult << "\t" << updateResult.updateIndexResult << "\t" << updateResult.updateTableResult << "\t"
              << updateResult.addPointsElapsed << "\t" << updateResult.updateIndexElapsed << "\t" << updateResult.updateTableElapsed << "\t"
              << meanError << "\t" << updateElapsed << "\t" << queryElapsed << "\t" << qps << "\t" << coverage << std::endl;
          }
        }
      }
    }

    delete[] samplePoints.ptr();
    delete[] exactDists.ptr();
  }

  log.close();
}


int main(){
  run();
  return 0;
}
