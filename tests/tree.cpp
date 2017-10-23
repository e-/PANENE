#include <catch.hpp>
#include <progressive_kd_tree_index.h>
#include <data_source/random_data_source.h>
#include <cstdlib>

using namespace panene;

typedef size_t IDType;
typedef float ElementType;
typedef float DistanceType;
using RandomSource = RandomDataSource<IDType, L2<ElementType>>;
using NeighborType = Neighbor<IDType, ElementType>;

void getExactNNL2(RandomSource &dataset, std::vector<std::vector<ElementType>> &queries, std::vector<std::vector<NeighborType>> &results, size_t k) {
  size_t n = dataset.size();
  size_t d = dataset.dim();

  for (size_t q = 0; q < queries.size(); q++) {
    std::vector<ElementType> &query = queries[q];
    std::vector<NeighborType> &neighbors = results[q];

    neighbors.resize(k);
    
    neighbors[0].id = 0;
    neighbors[0].dist = 0;
    
    for (size_t j = 0; j < d; ++j) neighbors[0].dist += (dataset.get(0, j) - query[j]) * (dataset.get(0, j) - query[j]);
    neighbors[0].dist = sqrt(neighbors[0].dist);

    size_t dcnt = 1;

    for (size_t i = 1; i < n; ++i) {
      DistanceType tmp = 0;
      for (size_t j = 0; j < d; ++j) tmp += (dataset.get(i, j) - query[j]) * (dataset.get(i, j) - query[j]);
      tmp = sqrt(tmp);
      
      if (dcnt < k) {
        neighbors[dcnt].id = i;
        neighbors[dcnt].dist = tmp;
        dcnt++;
      }
      else if (tmp < neighbors[dcnt - 1].dist) {
        neighbors[dcnt - 1].id = i;
        neighbors[dcnt - 1].dist = tmp;
      }

      size_t j = dcnt - 1;
      // bubble up
      while (j >= 1 && neighbors[j].dist < neighbors[j - 1].dist) {
        std::swap(neighbors[j], neighbors[j - 1]);
        j--;
      }
    }
  }
}

TEST_CASE("k-nearest neighbors should be found with correct distances in ascending order", "[KNN]") {
  srand(100);

  const size_t n = 100;
  const size_t d = 10;
  const size_t k = 10;

  RandomSource randomDataSource(n, d);
  const IndexParams indexParam(4);

  ProgressiveKDTreeIndex<RandomSource> progressiveIndex(indexParam, TreeWeight(1, 0));
  progressiveIndex.setDataSource(&randomDataSource);

  const size_t query_n = 100;
  std::vector<std::vector<ElementType>> queries(query_n);

  for(size_t i = 0; i < query_n; ++i) {
    queries[i].resize(d);

    for(size_t j = 0; j < d; ++j) {
      queries[i][j] = static_cast <ElementType> (rand()) / static_cast <ElementType>(RAND_MAX);
    }
  }

  progressiveIndex.run(n); // insert all points
  SearchParams searchParam(n * 2); // search for a sufficient number of points

  std::vector<ResultSet<IDType, ElementType>> results(query_n);
  for (size_t i = 0; i < query_n; ++i)
    results[i] = ResultSet<IDType, ElementType>(k);

  progressiveIndex.knnSearch(queries, results, k, searchParam);

  std::vector<std::vector<NeighborType>> exact_results(query_n);
  getExactNNL2(randomDataSource, queries, exact_results, k);

  size_t q = 0;
  for(auto& neighbors: results) {
    for(size_t i = 0 ; i < k; ++i) {
      // find exact neighbors
      REQUIRE(neighbors[i].id == exact_results[q][i].id);

      // with correct distances
      REQUIRE(neighbors[i].dist == Approx(exact_results[q][i].dist));
    }
    for(size_t i = 0; i < k-1; ++i) {
      // in ascending order
      REQUIRE(neighbors[i].dist <= neighbors[i+1].dist);
    }
    q++;
  }
}

TEST_CASE("masked points should not appear", "[KNN]") {
  srand(42);

  const size_t n = 100;
  const size_t d = 10;
  const size_t k = 10;

  RandomSource randomDataSource(n, d);
  const IndexParams indexParam(4);

  ProgressiveKDTreeIndex<RandomSource> progressiveIndex(indexParam, TreeWeight(1, 0));
  progressiveIndex.setDataSource(&randomDataSource);

  const size_t query_n = 100;
  std::vector<std::vector<ElementType>> queries(query_n);

  for(size_t i = 0; i < query_n; ++i) {
    queries[i].resize(d);

    for(size_t j = 0; j < d; ++j) {
      queries[i][j] = static_cast <ElementType> (rand()) / static_cast <ElementType>(RAND_MAX);
    }
  }

  progressiveIndex.run(n); // insert all points
  SearchParams searchParam(n * 2); // search for a sufficient number of points

  Roaring *mask = new Roaring();
  for(uint32_t i = 0; i < n; ++i) {
    if(rand() < RAND_MAX / 2)
      mask->add(i);
  }

  searchParam.mask = mask;

  std::vector<ResultSet<IDType, ElementType>> results(query_n);
  for (size_t i = 0; i < query_n; ++i)
    results[i] = ResultSet<IDType, ElementType>(k);

  progressiveIndex.knnSearch(queries, results, k, searchParam);

  std::vector<std::vector<NeighborType>> exact_results(query_n);
  getExactNNL2(randomDataSource, queries, exact_results, k);

  for(auto& neighbors: results) {
    for(size_t i = 0 ; i < k; ++i)
      REQUIRE_FALSE(mask->contains(neighbors[i].id));
  }

  delete mask;
}
