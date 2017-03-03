#include "paknn.h"
#include <memory>
#include <fstream>

using namespace flann;
using namespace std;

void readData(const string &path, int rows, int cols, float *dataset) {
  std::ifstream ifs(path);

  for(int i = 0; i < rows * cols; ++i) {
    ifs >> dataset[i]; 
  }
}

int main(){
  int n = 5000; // rows per each chunk
  int d = 100;
  int repeat = 10;
  int k = 10;

  float *data = new float[n * d * repeat];
  readData("../data/glove.shuffled.txt", n * repeat, d, data);
  SearchParams searchParam(512);

  KNNTable<Index<L2<float>>> table(k + 1, d, KDTreeIndexParams(4), searchParam);

  for(int r = 0; r < repeat; ++r) { 
    Matrix<float> dataMatrix(data + r * n, n, d);
    table.addPoints(dataMatrix);
  }

  // for testing
  
  int id = 5;
  for(auto& neighbor : table.getNeighbors(id)) { 
    cout << neighbor.id << ' ' << neighbor.distance << endl;
  }

  delete[] data;

  return 0;
}
