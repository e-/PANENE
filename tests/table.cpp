#include <catch.hpp>
#include <progressive_kd_tree_index.h>
#include <progressive_knn_table.h>
#include <data_source/random_data_source.h>
#include <data_sink/vector_data_sink.h>
#include <cstdlib>
#include <cmath>
#include <iostream>

using namespace panene;

typedef size_t IDType;
typedef float ElementType;
typedef float DistanceType;
using Source = RandomDataSource<IDType, L2<ElementType>>;
using Sink = VectorDataSink<IDType, DistanceType>;

TEST_CASE("Table using a given indexer", "[KNN]") {
  srand(100);

  const size_t n = 100;
  const size_t d = 10;
  const size_t k = 10;

  Source randomDataSource(n, d);
  Sink dataSink(n, k);

  const IndexParams indexParam(4);
  const SearchParams searchParam(30);

  ProgressiveKDTreeIndex<Source> indexer(&randomDataSource, indexParam, TreeWeight(1, 0));
  ProgressiveKNNTable<ProgressiveKDTreeIndex<Source>, Sink> table(&indexer, &dataSink, k, searchParam, TableWeight(0, 1));

  size_t batch = 20;

  for(size_t i = 0; i < n / batch; ++i) {
    auto indexerResult = indexer.run(batch);
    auto tableResult = table.runWithoutIndexer(indexerResult, batch);
   
    REQUIRE(indexerResult.numPointsInserted == batch * (i + 1)); 
    REQUIRE(tableResult.numPointsInserted == batch * (i + 1));

    std::cerr << indexerResult << std::endl;
    std::cerr << tableResult << std::endl;
  } 
}
