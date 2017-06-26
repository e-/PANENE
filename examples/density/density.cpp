#include <progressive_knn_table.h>
#include <scheduler/linear_scheduler.h>
#include <naive_data_source.h>
#include <util/timer.h>
#include <fstream>

using namespace panene;

int main(){
  NaiveDataSource dataSource;
  size_t k = 20;
  size_t n = 300000;
  size_t d = 2;
  SearchParams searchParam(512);
  searchParam.cores = 1;
  int maxOps = 10000;
  int bins = 256;

  Timer timer;
  float xMin, yMin, xMax, yMax, xStep, yStep;

  dataSource.open("data.sorted.txt", n, d);
  
  xMin = xMax = dataSource.get(0, 0);
  yMin = yMax = dataSource.get(0, 1);

  for(size_t i = 0; i < n; ++i) {
    if(xMin > dataSource.get(i, 0)) xMin = dataSource.get(i, 0);
    if(xMax < dataSource.get(i, 0)) xMax = dataSource.get(i, 0);
    if(yMin > dataSource.get(i, 1)) yMin = dataSource.get(i, 1);
    if(yMax < dataSource.get(i, 1)) yMax = dataSource.get(i, 1);
  }
    
  xStep = (xMax - xMin) / (bins - 1);
  yStep = (yMax - yMin) / (bins - 1);

  ProgressiveKNNTable<ProgressiveKDTreeIndex<L2<float>, NaiveDataSource>, NaiveDataSource> table(k + 1, d, IndexParams(4), searchParam);
  
  table.setDataSource(&dataSource);
  table.setScheduler(new LinearScheduler(1, 1, 0)); // we will not use the table, so set the weight for updating the table to zero.
  
  std::vector<std::vector<float>> samplePoints;

  for(int i = 0; i < bins; ++i) {
    for(int j = 0; j < bins; ++j) {
      float x = xMin + xStep * i, y = yMin + yStep * j;
      std::vector<float> point = {x, y};

      samplePoints.push_back(point);
    }
  }

  for(int r = 0; r <= 100; ++r) {
    timer.begin();
    auto result = table.update(maxOps);
    double elapsed = timer.end();

    std::cerr << r << "\t" << result.addNewPointResult << "\t" 
        << result.updateIndexResult << "\t" 
        << result.updateTableResult << "\t" 
        << elapsed << std::endl;
    
    // density estimation for sample points
    // Ultimately, this step needs to be progressive, but in this example we compute it sequentially for simplicity[]
    
    if(r % 5 > 0) continue;

    std::vector<ResultSet<size_t, float>> resultSets(bins * bins);
    for(size_t i = 0; i < bins * bins; ++i) {
      resultSets[i] = ResultSet<size_t, float>(k + 1);
    }

    table.indexer.knnSearch(samplePoints, resultSets, k, searchParam);

    std::ofstream ofs("progressive." + std::to_string(r / 5) + ".csv");

    ofs << "index,x,y,p" << std::endl;

    for(int i = 0; i < bins; ++i) {
      for(int j = 0; j < bins; ++j) {
        int index = i * bins + j;
        float r = resultSets[index][k].dist;
        float p = k / r / r / 3.141592f;
      
        ofs << index << "," << i << "," << j << "," << p << std::endl;
      }
    }  
  }


  return 0;
}
