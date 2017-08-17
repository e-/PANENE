#ifndef panene_progressive_kd_tree_index_h
#define panene_progressive_kd_tree_index_h

#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <queue>
#include <cassert>
#include <map>

#include <base_index.h>

namespace panene
{

template <typename Distance, typename DataSource>
class ProgressiveKDTreeIndex : public BaseIndex<Distance, DataSource>
{
public:

  enum UpdateStatus {
    NoUpdate,
    BuildingTree,
    InsertingPoints
  };

public:

  ProgressiveKDTreeIndex(IndexParams indexParams_, Distance distance_ = Distance()) : BaseIndex<Distance, DataSource>(indexParams_, distance_) {
  }

  size_t addPoints(size_t newPoints) {
    size_t oldSize = size;
    size += newPoints;
    if(size > dataSource -> loaded())
      size = dataSource -> loaded();

    if(oldSize == 0) { // for the first time, build the index as we did in the non-progressive version.
      buildIndex();
      return newPoints;
    }
    else {
      for(size_t i = oldSize; i < size; ++i) {
        for(size_t j = 0; j < numTrees; ++j) {
          trees[j]->size++;
          addPointToTree(trees[j], trees[j] -> root, i, 0);
        }
      }

      if (updateStatus == UpdateStatus::InsertingPoints) {
        for (size_t i = oldSize; i < size && sizeAtUpdate < size; ++i) {
          ongoingTree->size++;
          addPointToTree(ongoingTree, ongoingTree->root, sizeAtUpdate++, 0);
        }
      }
      return size - oldSize;
    }
  }
  
  void beginUpdate() {
    updateStatus = UpdateStatus::BuildingTree;
    sizeAtUpdate = size;
    ids.resize(sizeAtUpdate);

    for (size_t i = 0; i < sizeAtUpdate; ++i) ids[i] = int(i);
    std::random_shuffle(ids.begin(), ids.end());

    ongoingTree = new KDTree<NodePtr>();
    ongoingTree->root = new(pool) Node();
    std::queue<NodeSplit> empty;
    queue = empty;
    queue.push(NodeSplit(ongoingTree->root, &ids[0], sizeAtUpdate, 1));

    ongoingTree->size = sizeAtUpdate;
    ongoingTree->setMaxSize(dataSource->size());
  }

  size_t update(int ops) {
    int updatedCount = 0;

    while((ops == -1 || updatedCount < ops) && !queue.empty()) {
      NodeSplit nodeSplit = queue.front();
      queue.pop(); 

#if DEBUG
      std::cerr << "updatedCount " << updatedCount << std::endl;
#endif

      NodePtr node = nodeSplit.node;
      IDType *begin = nodeSplit.begin;
      int count = nodeSplit.count;
      int depth = nodeSplit.depth;

#if DEBUG
      std::cerr << begin << " " << count << std::endl;
#endif

      // At this point, nodeSplit the two children of nodeSplit are nullptr
      if (count == 1) {
        node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
        node->id = *begin;    /* Store index of this vec. */ // TODO id of vec
        ongoingTree->setInsertionLog(node->id, 1, depth);
      }
      else {
        int idx;
        int cutfeat;
        DistanceType cutval;
        meanSplit(begin, count, idx, cutfeat, cutval);

#if DEBUG        
        std::cerr << "cut index: " << idx << " cut count: " << count << std::endl;
#endif

        node->divfeat = cutfeat;
        node->divval = cutval;
        node->child1 = new(pool) Node();
        node->child2 = new(pool) Node();
        
        queue.push(NodeSplit(node->child1, begin, idx, depth + 1));
        queue.push(NodeSplit(node->child2, begin + idx, count - idx, depth + 1));
      }
      updatedCount++;
    }

    if (updateStatus == UpdateStatus::BuildingTree && queue.empty()) {
      updateStatus = UpdateStatus::InsertingPoints;
    }

    if (updateStatus == UpdateStatus::InsertingPoints) {
      if (ongoingTree->size < size) {
        // insert points from sizeAtUpdate to size

        while (ongoingTree->size < size && (ops == -1 || updatedCount < ops)) {
          ongoingTree->size++;
          addPointToTree(ongoingTree, ongoingTree->root, sizeAtUpdate, 0);

          sizeAtUpdate++;
          updatedCount++;
        }
      }

      if (ongoingTree->size >= size) {
        // finished creating a new tree
        ongoingTree->cost = ongoingTree->computeCost();

        size_t victimId = 0;
        float maxImbalance = trees[0]->computeImbalance();

        // find the most unbalanced one
        for (size_t i = 1; i < numTrees; ++i) {
          float imbalance = trees[i]->computeImbalance();

          if (maxImbalance < imbalance) {
            maxImbalance = imbalance;
            victimId = i;
          }
        }

        // get the victim
        auto victim = trees[victimId];

        // replace the victim with the newly created tree
        delete victim;

        trees[victimId] = ongoingTree;

        // reset the sizeAtUpdate
        sizeAtUpdate = 0;
        updateStatus = UpdateStatus::NoUpdate;
        queryLoss = 0;
      }
    }

    return updatedCount;
  }  

  void accumulateLoss(size_t n) {
    if (updateStatus == UpdateStatus::NoUpdate) {
      float lossDelta = 0;

      for (size_t i = 0; i < numTrees; ++i) {
        lossDelta += n * (trees[i]->getCachedCost() - std::log2(size));
      }
      queryLoss += lossDelta;

      float updateCost = std::log2(size) * size;

      if (queryLoss > updateCost * updateCostWeight) {
        beginUpdate();
      }
    }
  }

  void knnSearch(
    const IDType &qid,
    std::vector<ResultSet<IDType, DistanceType>> &resultSets,
    size_t knn,
    const SearchParams& params)
  {
    std::vector<std::vector<ElementType>> vectors(1);

    vectors[0].resize(dim);
    for (size_t i = 0; i < dim; ++i)
      vecdtors[0][i] = dataSoruce->get(qid, i);

    knnSearch(qid, resultsSets, knn, params);
  }

  void knnSearch(
      const std::vector<std::vector<ElementType>> &vectors,
      std::vector<ResultSet<IDType, DistanceType>> &resultSets,
      size_t knn,
      const SearchParams& params)
  {
    accumulateLoss(vectors.size());

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
    std::vector<IDType> ids(size);
    for(size_t i = 0; i < size; ++i) {
      ids[i] = IDType(i);
    }

    for(size_t i = 0; i < numTrees; ++i) {
      std::random_shuffle(ids.begin(), ids.end());
      trees[i]->root = divideTree(trees[i], &ids[0], size, 1);
      trees[i]->size = size;
      trees[i]->cost = trees[i]->computeCost();
    }
  }
    
  void addPointToTree(KDTree<NodePtr>* tree, NodePtr node, IDType id, int depth) {
    if ((node->child1==NULL) && (node->child2==NULL)) {
      // if leaf

      size_t nodeId = node->id;
      size_t divfeat = dataSource->findDimWithMaxSpan(id, nodeId);
      
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

  void freeIndex() {
    for (size_t i = 0; i < numTrees; ++i) {
      if (trees[i] != nullptr) trees[i]->~KDTree();
    }
    pool.free();
  }

 public: 
  float getMaxCachedCost() {
    float cost = 0;
    for (size_t i = 0; i < numTrees; ++i) {
      if (cost < trees[i]->getCachedCost()) {
        cost = trees[i]->getCachedCost();
      }
    }

    return cost;
  }

  std::vector<float> getCachedImbalances() {
    std::vector<float> imbalances;
    for(size_t i = 0; i < numTrees; ++i) {
      imbalances.push_back(trees[i]->getCachedImbalance());
    }
    return imbalances;
  }

  std::vector<float> recomputeImbalances() {
    std::vector<float> imbalances;

    for(size_t i = 0; i < numTrees; ++i) {
      imbalances.push_back(trees[i]->computeImbalance());
    }
    
    return imbalances;
  }

  size_t computeMaxDepth() {
    size_t maxDepth = 0;
    for(size_t j = 0; j < numTrees; ++j) {
      size_t depth = trees[j]->computeMaxDepth();
      if(maxDepth < depth)
        maxDepth = depth;
    }
    return maxDepth;
  }

  std::map<size_t, size_t> computeCountDistribution() {
    std::map<size_t, size_t> dict;
    for(size_t i = 0 ; i < numTrees; ++i) {
      const auto& partial = trees[i]->computeCountDistribution();
      for(const auto& tuple : partial) {
        if(dict.count(tuple.first) == 0)
          dict[tuple.first] = 0;
        dict[tuple.first] += tuple.second;
      }
    }
    return dict;
  }  

  void printBackstage() {
    std::cout << "queue size: " << queue.size() << std::endl;
    std::cout << "ongoingTree size: " << ongoingTree->size << std::endl;
  }

private:
  const float updateCostWeight = 0.25; // lower -> more update

  size_t sizeAtUpdate = 0;

  std::queue<NodeSplit> queue;
  std::vector<size_t> ids;

public:
  UpdateStatus updateStatus = UpdateStatus::NoUpdate;
  KDTree<NodePtr>* ongoingTree;
  float queryLoss = 0.0;
};

}

#endif
