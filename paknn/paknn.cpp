#include "paknn.h"
#include <memory>

using namespace flann;
using namespace std;

int main(){
  int k = 5;
  int d = 5;

  vector<float> values;
  
  for(int i = 0; i < 50; ++i) 
    values.push_back((float)i / 50);

  Matrix<float> data(&values[0], 50, d);
  SearchParams searchParam(512);

  KNNTable<Index<L2<float>>> table(k, d, KDTreeIndexParams(1), searchParam);

  for(int r = 0; r < 5; ++r) { 
    table.addPoints(data);
  }


  return 0;
}
