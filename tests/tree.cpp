#include <catch.hpp>
#include <progressive_kd_tree_index.h>
#include <data_source/random_data_source.h>
#include <data_source/vector_data_source.h>
#include <data_source/array_data_source.h>
#include <cstdlib>
#include <cmath>

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

  ProgressiveKDTreeIndex<RandomSource> progressiveIndex(&randomDataSource, indexParam, TreeWeight(1, 0));

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

  ProgressiveKDTreeIndex<RandomSource> progressiveIndex(&randomDataSource, indexParam, TreeWeight(1, 0));

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
  // randomly mask a half of points
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
    // check whether masked points do not appear
    for(size_t i = 0 ; i < k; ++i)
      REQUIRE_FALSE(mask->contains(neighbors[i].id));
  }

  delete mask;
}

using ArraySource = ArrayDataSource<IDType, L2<ElementType>>;

TEST_CASE("incremental imbalance updates", "[KNN]") {
  srand(42);

  /* what to test?
   * build 
   * check insertionLog (freq, depth)
   * search
   * check insertionLog
   * checkCostSum 
   */

  // prepare data with 5 points
  float points[14] = {
    1, 1,
    2, 3,
    3, 2,
    4, 4,
    4.5, 1.5,
    1.5, 5,
    3.6, 4.5
  };

  const size_t n = 7;
  const size_t d = 2;
  const size_t k = 2;

  ArraySource arrayDataSource(n, d, points);

  // create only one tree without random dim selection
  const IndexParams indexParam(1, 1);

  ProgressiveKDTreeIndex<ArraySource> progressiveIndex(&arrayDataSource, indexParam, TreeWeight(1, 0), 1000); 

  progressiveIndex.run(5); //insert the first five points

  auto& tree = progressiveIndex.trees[0];

  REQUIRE(tree->root->divfeat == 0);
  REQUIRE(tree->root->divval == 2.9f);

  REQUIRE(tree->root->child1->divfeat == 1);
  REQUIRE(tree->root->child1->divval == 2.0f);

  REQUIRE(tree->root->child2->divfeat == 1);
  REQUIRE(tree->root->child2->divval == 2.5f);
  
  for(size_t i=0;i<5;++i) {
    REQUIRE(tree->insertionLog[i].freq == 1);
  }

  REQUIRE(tree->insertionLog[0].depth == 3);
  REQUIRE(tree->insertionLog[1].depth == 3);
  REQUIRE(tree->insertionLog[2].depth == 4);
  REQUIRE(tree->insertionLog[3].depth == 3);
  REQUIRE(tree->insertionLog[4].depth == 4);

  float initialCost = (3.0f * 3 + 2.0f * 4) / 5;
  REQUIRE(tree->getCachedCost() == Approx(initialCost));
  
  // when search, is freq updated correctly?
  
  std::vector<std::vector<ElementType>> q1;
  std::vector<ElementType> p1 = {1.4f, 1.9f};
  q1.push_back(p1);
  
  SearchParams searchParam(100); // search for a sufficient number of points

  std::vector<ResultSet<IDType, ElementType>> result(1);
  result[0] = ResultSet<IDType, ElementType>(k);

  progressiveIndex.knnSearch(q1, result, 1, searchParam);
  
  REQUIRE(result[0][0].id == 0);

  // then, are the freq of points updated correctly?
  REQUIRE(tree->insertionLog[0].freq == 2);
  REQUIRE(tree->insertionLog[1].freq == 2);
  REQUIRE(tree->insertionLog[2].freq == 1);
  REQUIRE(tree->insertionLog[3].freq == 1);
  REQUIRE(tree->insertionLog[4].freq == 1);
  
  REQUIRE(tree->insertionLog[0].depth == 3);
  REQUIRE(tree->insertionLog[1].depth == 3);
  REQUIRE(tree->insertionLog[2].depth == 4);
  REQUIRE(tree->insertionLog[3].depth == 3);
  REQUIRE(tree->insertionLog[4].depth == 4);

  // is the cost updated correctly?
  float cost1 = (float)(2 * 3 + 2 * 3 + 1 * 4 + 1 * 3 + 1 * 4) / 7;
  REQUIRE(tree->getCachedCost() == Approx(cost1));
  
  // when insert new points, freq and depth updated correctly?
  /* new points:
    1.5, 5,
    3.6, 4.5 */
  
  progressiveIndex.run(2);

  REQUIRE(tree->root->child1->child2->divfeat == 1);
  REQUIRE(tree->root->child1->child2->divval == Approx(4.0f));

  REQUIRE(tree->root->child2->child2->divfeat == 1);
  REQUIRE(tree->root->child2->child2->divval == Approx(4.25f));
  
  REQUIRE(tree->insertionLog[0].freq == 2);
  REQUIRE(tree->insertionLog[1].freq == 3);
  REQUIRE(tree->insertionLog[2].freq == 1);
  REQUIRE(tree->insertionLog[3].freq == 2);
  REQUIRE(tree->insertionLog[4].freq == 1);
  REQUIRE(tree->insertionLog[5].freq == 1);
  REQUIRE(tree->insertionLog[6].freq == 1);
  
  REQUIRE(tree->insertionLog[0].depth == 3);
  REQUIRE(tree->insertionLog[1].depth == 4);
  REQUIRE(tree->insertionLog[2].depth == 4);
  REQUIRE(tree->insertionLog[3].depth == 4);
  REQUIRE(tree->insertionLog[4].depth == 4);
  REQUIRE(tree->insertionLog[5].depth == 4);
  REQUIRE(tree->insertionLog[6].depth == 4);

  // then, is the returned cost correct?
  float cost2 = (float)(2 * 3 + 3 * 4 + 1 * 4 + 2 * 4 + 1 * 4 + 1 * 4 + 1 * 4) / 11;
  REQUIRE(tree->getCachedCost() == Approx(cost2));
}
