#include "data/naive_data_source.h"
#include "test/tests.h"

int main(){
  panene::NaiveDataSource naiveDataSource;
  
  naiveDataSource.open("../data/sift.shuffled.txt", 100000, 128);

  panene::ProgressiveKNNTableBenchmark test(&naiveDataSource);

  test.run();
  return 0;
}
