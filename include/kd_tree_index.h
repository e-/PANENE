#ifndef panene_kd_tree_index_h
#define panene_kd_tree_index_h

#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <queue>
#include <cassert>

#include <base_index.h>

namespace panene
{

template <typename Distance, typename DataSource>
class KDTreeIndex : public BaseIndex<Distance, DataSource>
{
  USE_BASECLASS_SYMBOLS


public:
  KDTreeIndex(IndexParams indexParams_, Distance distance_ = Distance()) : BaseIndex<Distance, DataSource>(indexParams_, distance_) {
  }

  size_t addPoints(size_t newPoints) {
    size_t oldSize = size;
    size += newPoints;

    if(size > dataSource -> loaded())
      size = dataSource -> loaded();
    
    if(sizeAtBuild * 2 < size) {
      buildIndex();
    }
    else {
      for(size_t i = oldSize; i < size; ++i) {
        for(int j = 0; j < numTrees; ++j) {
          addPointToTree(trees[j], trees[j] -> root, i, 0);
        }
      }
    }
    return size - oldSize;
  }

  size_t update(int ops) {
    throw;
  }

  void knnSearch(
      const IDType &qid,
      ResultSet<IDType, DistanceType> &resultSets,
      size_t knn,
      const SearchParams& params) const
  {
    std::vector<std::vector<ElementType>> vectors(1);

    vectors[0].resize(dim);
    for (size_t i = 0; i < dim; ++i)
      vectors[0][i] = dataSource->get(qid, i);

    knnSearch(vectors, resultSets, knn, params);
  }

  void knnSearch(
      const std::vector<std::vector<ElementType>> &vectors,
      std::vector<ResultSet<IDType, DistanceType>> &resultSets,
      size_t knn,
      const SearchParams& params) const
  {
    
#pragma omp parallel num_threads(params.cores)
    {
#pragma omp for schedule(static)
      for (int i = 0; i < (int)vectors.size(); i++) {
        findNeighbors(vectors[i], resultSets[i], params);
        //ids_to_ids(ids[i], ids[i], n);
      }
    }
  }

protected:
  
  void buildIndex() {
    sizeAtBuild = size;
    std::vector<IDType> ids(size);
    for(size_t i = 0; i < size; ++i) {
      ids[i] = IDType(i);
    }

    trees.resize(numTrees);

    freeIndex();
    for(int i = 0; i < numTrees; ++i) {
      std::random_shuffle(ids.begin(), ids.end());            
      trees[i] = new KDTree<NodePtr>();
      trees[i]->setMaxSize(dataSource->size());
      trees[i]->root = divideTree(trees[i], &ids[0], size, 1);
      trees[i]->size = size;
      trees[i]->cost = trees[i]->computeCost();
    }
  }

  void freeIndex() {
    for(size_t i=0; i < numTrees; ++i) {
      if(trees[i] != nullptr) trees[i]->~KDTree();
    }
    pool.free();
  }
  
  void addPointToTree(KDTree<NodePtr>* tree, NodePtr node, IDType id, int depth) {
    if ((node->child1==NULL) && (node->child2==NULL)) {

      size_t nodeId = node->id;
      size_t divfeat = dataSource->findDimWithMaxSpan(id, node->id);
      
      NodePtr left = new(pool) Node();
      left->child1 = left->child2 = NULL;

      NodePtr right = new(pool) Node();
      right->child1 = right->child2 = NULL;

      ElementType pointValue = dataSource -> get(id, divfeat);
      ElementType leafValue = dataSource -> get(node->id, divfeat);

      if (pointValue < leafValue) {
          left->id = id;
          right->id = node->id;
      }
      else {
          left->id = node->id;
          right->id = id;
      }
      
      left->divfeat = right->divfeat = -1;

      node->divfeat = divfeat;
      node->divval = (pointValue + leafValue)/2;
      node->child1 = left;
      node->child2 = right;            

      // incrementally update imbalance      
      tree->setInsertionLog(id, 0, depth + 1);
      tree->markSplit(nodeId);
    }
    else {
      if (dataSource->get(id, node->divfeat) < node->divval) {
        addPointToTree(tree, node->child1, id, depth + 1);
      }
      else {
        addPointToTree(tree, node->child2, id, depth + 1);
      }
    }
  }
  
public:
  std::vector<float> getCachedImbalances() {
    std::vector<float> imbalances;
    for (size_t i = 0; i < numTrees; ++i) {
      imbalances.push_back(trees[i]->getCachedImbalance());
    }
    return imbalances;
  }

  std::vector<float> recomputeImbalances() {
    std::vector<float> imbalances;

    for (size_t i = 0; i < numTrees; ++i) {
      imbalances.push_back(trees[i]->computeImbalance());
    }

    return imbalances;
  }

  size_t computeMaxDepth() {
    size_t maxDepth = 0;
    for (size_t j = 0; j < numTrees; ++j) {
      size_t depth = trees[j]->computeMaxDepth();
      if (maxDepth < depth)
        maxDepth = depth;
    }
    return maxDepth;
  }

  std::map<size_t, size_t> computeCountDistribution() {
    std::map<size_t, size_t> dict;
    for (size_t i = 0; i < numTrees; ++i) {
      const auto& partial = trees[i]->computeCountDistribution();
      for (const auto& tuple : partial) {
        if (dict.count(tuple.first) == 0)
          dict[tuple.first] = 0;
        dict[tuple.first] += tuple.second;
      }
    }
    return dict;
  }

private:
  size_t sizeAtBuild = 0;
};

}
#endif
