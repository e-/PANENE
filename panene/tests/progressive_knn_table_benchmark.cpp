#include "progressive_knn_table_benchmark.h"

int main(){
  panene::NaiveDataSource naiveDataSource;
  
  naiveDataSource.open("../../../data/sift.shuffled.txt", 100000, 128);

  panene::ProgressiveKNNTableBenchmark test(&naiveDataSource);

  test.run();
  return 0;
}
