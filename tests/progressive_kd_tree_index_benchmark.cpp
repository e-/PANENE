#include <iostream>

#include <progressive_knn_table.h>
#include <binary_data_source.h>
#include <util/timer.h>
#include <util/matrix.h>

int main() {
  panene::BinaryDataSource trainDataSource;
  panene::BinaryDataSource testDataSource;

  trainDataSource.open("../../data/glove/train.bin", 100000, 100);
  testDataSource.open("../../data/glove/test.bin", 100, 100);

  std::cout << trainDataSource.get(1, 1) << std::endl;
  return 0;
}
