#include "util/data_source.h"
#include "test/tests.h"

int main(){
  DataSource dataSource("../data/sift.shuffled.txt", "sift", 100000, 128);

  ProgressiveKNNTableBenchmark test(dataSource);

  test.run();
  return 0;
}
