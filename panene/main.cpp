#include <memory>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <flann/flann.hpp>

#include "test/tests.h"

int main(){
  ProgressiveKNNTableTest test;

  test.run();
  return 0;
}
