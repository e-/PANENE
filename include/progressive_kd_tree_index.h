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

#include <progressive_base_index.h>
#include <kd_tree.h>

namespace panene
{

template <typename Distance, typename DataSource>
class ProgressiveKDTreeIndex : public ProgressiveBaseIndex<Distance, DataSource>
{
public:
  typedef typename DataSource::ElementType ElementType;
  typedef typename DataSource::DistanceType DistanceType;
  typedef typename DataSource::IDType IDType;

  enum UpdateStatus {
    NoUpdate,
    BuildingTree,
    InsertingPoints
  };

protected:
  struct Node {
    int divfeat;
    DistanceType divval;
    //ElementType* point;
    IDType id; // point id for a leaf node, 0 otherwise

    Node *child1, *child2;

    Node() {
      child1 = child2 = nullptr;
    }

    ~Node() {
      if(child1 != nullptr) {child1->~Node(); child1 = nullptr;}
      if(child2 != nullptr) {child2->~Node(); child2 = nullptr;}
    }
  };
  
  struct NodeSplit { 
    struct Node *node;
    IDType *begin;
    int count;
    int depth;
    
    NodeSplit(Node* node_, IDType *begin_, int count_, int depth_) : node(node_), begin(begin_), count(count_), depth(depth_) {}
  };

  template <typename T, typename DistanceType>
  struct BranchStruct
  {
    T node;           /* Tree node at which search resumes */
    DistanceType mindist;     /* Minimum distance to query for all nodes below. */

    BranchStruct() {}
    
    BranchStruct(const T& aNode, DistanceType dist) : node(aNode), mindist(dist) {}

    bool operator<(const BranchStruct<T, DistanceType>& rhs) const
    {
      return mindist<rhs.mindist;
    }

  }; 

  typedef Node* NodePtr;
  typedef BranchStruct<NodePtr, DistanceType> BranchSt;
  typedef BranchSt* Branch;  

public:
  ProgressiveKDTreeIndex(int numTrees_, Distance distance_ = Distance()): numTrees(numTrees_), distance(distance_)
  {
    trees.resize(numTrees);
    for(size_t i = 0; i < numTrees; ++i) {
      trees[i] = new KDTree<NodePtr>();
    }
  }

  ProgressiveKDTreeIndex(IndexParams indexParams_, Distance distance_ = Distance()): distance(distance_) {
    numTrees = indexParams_.trees;
    trees.resize(numTrees);
    for(size_t i = 0; i < numTrees; ++i) {
      trees[i] = new KDTree<NodePtr>();
    }
  }

  ~ProgressiveKDTreeIndex() {
    for(size_t i = 0; i < numTrees; ++i) {
      delete trees[i];
    }
  }

  void setDataSource(DataSource *dataSource_) {
    dataSource = dataSource_;
    dim = dataSource -> dim();
    size_t maxSize = dataSource -> size();
    for(size_t i = 0; i < numTrees; ++i) {
      trees[i]->setMaxSize(maxSize);
    }
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

  size_t getSize() { return size; }

  int usedMemory() const {
    return int(pool.usedMemory+pool.wastedMemory+size*sizeof(int));  // pool memory and vind array memory
  }

  void knnSearch(
      const std::vector<std::vector<ElementType>> &vectors,
      std::vector<ResultSet<IDType, DistanceType>> &resultSets,
      size_t knn,
      const SearchParams& params)
  {
    bool use_heap = false; // TODO

    // loss accumultation

    if (updateStatus == UpdateStatus::NoUpdate) {
      float lossDelta = 0;

      for (size_t i = 0; i < numTrees; ++i) {
        lossDelta += vectors.size() * (trees[i]->getCachedCost() - std::log2(size));
      }
      queryLoss += lossDelta;

      float updateCost = std::log2(size) * size;

      if (queryLoss > updateCost * updateCostWeight) {
        beginUpdate();
      }
    }    

    //std::cerr << queryLoss << " " << size << " " << size * std::log2(size)  << std::endl;

    /*if (params.use_heap==FLANN_Undefined) {
      use_heap = (knn>KNN_HEAP_THRESHOLD)?true:false;
    }
    else {
      use_heap = (params.use_heap==FLANN_True)?true:false;
    }*/

    if (use_heap) {
#pragma omp parallel num_threads(params.cores)
      {
#pragma omp for schedule(static)
        for (int i = 0; i < (int)vectors.size(); i++) {
          findNeighbors(vectors[i], resultSets[i], params);
          //ids_to_ids(ids[i], ids[i], n);
        }
      }
    }
    else {
#pragma omp parallel num_threads(params.cores)
      {
#pragma omp for schedule(static)
        for (int i = 0; i < (int)vectors.size(); i++) {
          findNeighbors(vectors[i], resultSets[i], params);
          //ids_to_ids(ids[i], ids[i], n);
        }
      }
    }
  }


  /**
   * Find set of nearest neighbors to vec. Their ids are stored inside
   * the result object.
   *
   * Params:
   *     result = the result object in which the ids of the nearest-neighbors are stored
   *     vec = the vector for which to search the nearest neighbors
   *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
   */
  void findNeighbors(const IDType &qid, ResultSet<IDType, DistanceType> &result, const SearchParams& searchParams) const
  {
    int maxChecks = searchParams.checks;
    float epsError = 1+searchParams.eps;

    if (maxChecks==0) { // }FLANN_CHECKS_UNLIMITED) {
      throw;
      //getExactNeighbors<false>(qid, result, epsError);
      // TODO deletion
      /*if (removed_) {
        getExactNeighbors<true>(result, vec, epsError);
      }
      else {
        getExactNeighbors<false>(result, vec, epsError);
      }*/
    }
    else {
      getNeighbors<false>(qid, result, maxChecks, epsError);
      // TODO deletion
      /*if (removed_) {
        getNeighbors<true>(result, vec, maxChecks, epsError);
      }
      else {
        getNeighbors<false>(result, vec, maxChecks, epsError);
      }*/
    }
  }

  void findNeighbors(const std::vector<ElementType> &vec, ResultSet<IDType, DistanceType> &result, const SearchParams& searchParams) const
  {
    int maxChecks = searchParams.checks;
    float epsError = 1+searchParams.eps;

    if (maxChecks==0) { // }FLANN_CHECKS_UNLIMITED) {
      throw;
      //getExactNeighbors<false>(vec, result, epsError);
      // TODO deletion
      /*if (removed_) {
        getExactNeighbors<true>(result, vec, epsError);
      }
      else {
        getExactNeighbors<false>(result, vec, epsError);
      }*/
    }
    else {
      getNeighbors<false>(vec, result, maxChecks, epsError);
      // TODO deletion
      /*if (removed_) {
        getNeighbors<true>(result, vec, maxChecks, epsError);
      }
      else {
        getNeighbors<false>(result, vec, maxChecks, epsError);
      }*/
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

  void freeIndex() {
    for(size_t i=0; i < numTrees; ++i) {
      if(trees[i] != nullptr) trees[i]->~KDTree();
    }
    pool.free();
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

  NodePtr divideTree(KDTree<NodePtr>* tree, IDType *ids, size_t count, size_t depth) {
    NodePtr node = new(pool) Node();
    
    /* If too few exemplars remain, then make this a leaf node. */
    if (count == 1) {
        node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
        node->divfeat = -1; // a leaf node
        IDType id = ids[0];
        node->id = id;
        tree->setInsertionLog(id, 1, depth);
    }
    else {
        int idx;
        int cutfeat;
        DistanceType cutval;
        meanSplit(ids, count, idx, cutfeat, cutval);

        node->divfeat = cutfeat;
        node->divval = cutval;
        node->child1 = divideTree(tree, ids, idx, depth + 1);
        node->child2 = divideTree(tree, ids+idx, count-idx, depth + 1);
    }

    return node;
  }

   /**
    * Choose which feature to use in order to subdivide this set of vectors.
    * Make a random choice among those with the highest variance, and use
    * its variance as the threshold value.
    **/
  void meanSplit(IDType *ids, int count, int &index, int &cutfeat, DistanceType &cutval) {
    /* Compute mean values.  Only the first SAMPLE_MEAN values need to be
        sampled to get a good estimate.
     */

    int sampleCount = std::min((int)SAMPLE_MEAN+1, count);
    std::vector<DistanceType> mean(dim), var(dim);
    
    dataSource->computeMeanAndVar(ids, sampleCount, mean, var);

    /* Select one of the highest variance ids at random. */
    cutfeat = selectDivision(var);
    cutval = mean[cutfeat];

    int lim1, lim2;
    planeSplit(ids, count, cutfeat, cutval, lim1, lim2);

    if (lim1>count/2) index = lim1;
    else if (lim2<count/2) index = lim2;
    else index = count/2;

    /* If either list is empty, it means that all remaining features
     * are identical. Split in the middle to maintain a balanced tree.
     */
    if ((lim1==count)||(lim2==0)) index = count/2;  
  }
  
  /**
   * Select the top RAND_DIM largest values from v and return the index of
   * one of these selected at random.
   */
  int selectDivision(const std::vector<DistanceType> &v)
  {
    int num = 0;
    size_t topind[RAND_DIM];

    /* Create a list of the ids of the top RAND_DIM values. */
    for (size_t i = 0; i < dim; ++i) {
      if ((num < RAND_DIM)||(v[i] > v[topind[num-1]])) {
        /* Put this element at end of topind. */
        if (num < RAND_DIM) {
          topind[num++] = i;            /* Add to list. */
        }
        else {
          topind[num-1] = i;         /* Replace last element. */
        }
        /* Bubble end value down to right location by repeated swapping. */
        int j = num - 1;
        while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
          std::swap(topind[j], topind[j-1]);
          --j;
        }
      }
    }
    /* Select a random integer in range [0,num-1], and return that index. */
    int rnd = rand_int(num); 
//    std::cerr << "rnd: " << rnd << "so chosen: " << topind[rnd] << std::endl;
    return (int)topind[rnd]; 
  }

  /**
   *  Subdivide the list of points by a plane perpendicular on axe corresponding
   *  to the 'cutfeat' dimension at 'cutval' position.
   *
   *  On return:
   *  dataset[ind[0..lim1-1]][cutfeat]<cutval
   *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
   *  dataset[ind[lim2..count]][cutfeat]>cutval
   */
  void planeSplit(IDType *ids, int count, int cutfeat, DistanceType cutval, int& lim1, int& lim2)
  {
    /* Move vector ids for left subtree to front of list. */
    int left = 0;
    int right = count-1;
    for (;; ) {
      while (left<=right && dataSource -> get(ids[left], cutfeat) < cutval) ++left; // TODO
      while (left<=right && dataSource -> get(ids[right], cutfeat) >= cutval) --right; // TODO
      if (left>right) break;
      std::swap(ids[left], ids[right]); ++left; --right;
    }
    lim1 = left;
    right = count-1;
    for (;; ) {
      while (left<=right && dataSource -> get(ids[left], cutfeat) <= cutval) ++left; // TODO
      while (left<=right && dataSource -> get(ids[right], cutfeat) > cutval) --right; // TODO
      if (left>right) break;
      std::swap(ids[left], ids[right]); ++left; --right;
    }
    lim2 = left;
  }

  /**
   * Performs an exact nearest neighbor search. The exact search performs a full
   * traversal of the tree.
   */
  template<bool with_removed>
  void getExactNeighbors(const IDType &qid, ResultSet<IDType, DistanceType> &result, float epsError) const
  {
    if (numTrees > 1) {
      fprintf(stderr,"It doesn't make any sense to use more than one tree for exact search");
    }
    if (numTrees > 0) {
      searchLevelExact<with_removed>(qid, result, trees[0]->root, 0.0, epsError);
    }
  }

  /**
   * Performs the approximate nearest-neighbor search. The search is approximate
   * because the tree traversal is abandoned after a given number of descends in
   * the tree.
   */
  template<bool with_removed>
  void getNeighbors(const IDType &qid, ResultSet<IDType, DistanceType> &result, int maxCheck, float epsError) const
  {
    BranchSt branch;

    int checkCount = 0;
    Heap<BranchSt>* heap = new Heap<BranchSt>((int)size);
    DynamicBitset checked(size);

    /* Search once through each tree down to root. */
    for (size_t i = 0; i < numTrees; ++i) {
      searchLevel<with_removed>(qid, result, trees[i]->root, 0, checkCount, maxCheck, epsError, heap, checked);
    }

    /* Keep searching other branches from heap until finished. */
    while ( heap->popMin(branch) && (checkCount < maxCheck || !result.full() )) {
      searchLevel<with_removed>(qid, result, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
    }

    delete heap;
  }

  template<bool with_removed>
  void getNeighbors(const std::vector<ElementType> &vec, ResultSet<IDType, DistanceType> &result, int maxCheck, float epsError) const
  {
    int i;
    BranchSt branch;

    int checkCount = 0;
    Heap<BranchSt>* heap = new Heap<BranchSt>((int)size);
    DynamicBitset checked(size);

    /* Search once through each tree down to root. */
    for (i = 0; i < numTrees; ++i) {
      searchLevel<with_removed>(vec, result, trees[i]->root, 0, checkCount, maxCheck, epsError, heap, checked);
    }

    /* Keep searching other branches from heap until finished. */
    while ( heap->popMin(branch) && (checkCount < maxCheck || !result.full() )) {
      searchLevel<with_removed>(vec, result, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
    }

    delete heap;
  }

  /**
   *  Search starting from a given node of the tree.  Based on any mismatches at
   *  higher levels, all exemplars below this level must have a distance of
   *  at least "mindistsq".
   */
  template<bool with_removed>
  void searchLevel(const IDType &qid, ResultSet<IDType, DistanceType> &result_set, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
      float epsError, Heap<BranchSt>* heap, DynamicBitset& checked) const
  {
    if (result_set.worstDist < mindist) {
      //      printf("Ignoring branch, too far\n");
      return;
    }

    /* If this is a leaf node, then do check and return. */
    if ((node->child1 == NULL)&&(node->child2 == NULL)) {
      int id = node -> id;
      // TODO
/*      if (with_removed) {
        if (removed_points_.test(index)) return;
      }*/
      /*  Do not check same node more than once when searching multiple numTrees. */
      if ( checked.test(id) || ((checkCount>=maxCheck)&& result_set.full()) ) return;
      checked.set(id);
      checkCount++;

      DistanceType dist = dataSource -> distL2Squared(id, qid);
      result_set << Neighbor<IDType, DistanceType>(id, dist);
      return;
    }

    /* Which child branch should be taken first? */
    ElementType val = dataSource->get(qid, node -> divfeat);
    DistanceType diff = val - node->divval;
    NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
    NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

    /* Create a branch record for the branch not taken.  Add distance
       of this feature boundary (we don't attempt to correct for any
       use of this feature in a parent node, which is unlikely to
       happen and would have only a small effect).  Don't bother
       adding more branches to heap after halfway point, as cost of
       adding exceeds their value.
       */

    DistanceType new_distsq = mindist + distance.accum_dist(val, node->divval, node->divfeat);
    //    if (2 * checkCount < maxCheck  ||  !result.full()) {
    if ((new_distsq*epsError < result_set.worstDist)||  !result_set.full()) {
      heap->insert( BranchSt(otherChild, new_distsq) );
    }

    /* Call recursively to search next level down. */
    searchLevel<with_removed>(qid, result_set, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
  }

  /**
   *  Search starting from a given node of the tree.  Based on any mismatches at
   *  higher levels, all exemplars below this level must have a distance of
   *  at least "mindistsq".
   */
  template<bool with_removed>
  void searchLevel(const std::vector<ElementType> &vec, ResultSet<IDType, DistanceType> &result_set, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
      float epsError, Heap<BranchSt>* heap, DynamicBitset& checked) const
  {
    if (result_set.worstDist < mindist) {
      //      printf("Ignoring branch, too far\n");
      return;
    }

    /* If this is a leaf node, then do check and return. */
    if ((node->child1 == NULL)&&(node->child2 == NULL)) {
      int id = node -> id;
      // TODO
/*      if (with_removed) {
        if (removed_points_.test(index)) return;
      }*/
      /*  Do not check same node more than once when searching multiple numTrees. */
      if ( checked.test(id) || ((checkCount>=maxCheck)&& result_set.full()) ) return;
      checked.set(id);
      checkCount++;

      DistanceType dist = DistanceType(0);
      
      for(size_t i = 0; i < dim; ++i) {
        ElementType x = dataSource -> get(id, i);
        ElementType y = vec[i];

        dist += (x - y) * (x - y);
      }
      result_set << Neighbor<IDType, DistanceType>(id, dist);
      return;
    }

    /* Which child branch should be taken first? */
    ElementType val = vec[node -> divfeat]; //dataSource->get(qid, node -> divfeat);
    DistanceType diff = val - node->divval;
    NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
    NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

    /* Create a branch record for the branch not taken.  Add distance
       of this feature boundary (we don't attempt to correct for any
       use of this feature in a parent node, which is unlikely to
       happen and would have only a small effect).  Don't bother
       adding more branches to heap after halfway point, as cost of
       adding exceeds their value.
       */

    DistanceType new_distsq = mindist + distance.accum_dist(val, node->divval, node->divfeat);
    //    if (2 * checkCount < maxCheck  ||  !result.full()) {
    if ((new_distsq*epsError < result_set.worstDist)||  !result_set.full()) {
      heap->insert( BranchSt(otherChild, new_distsq) );
    }

    /* Call recursively to search next level down. */
    searchLevel<with_removed>(vec, result_set, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
  }

  /**
   * Performs an exact search in the tree starting from a node.
   */
  template<bool with_removed>
  void searchLevelExact(const IDType &qid, ResultSet<IDType, DistanceType> &result_set, const NodePtr node, DistanceType mindist, const float epsError) const
  {
    /* If this is a leaf node, then do check and return. */
    if ((node->child1 == NULL)&&(node->child2 == NULL)) {
      IDType id = node->id;
      // TODO
/*      if (with_removed) {
        if (removed_points_.test(index)) return; // ignore removed points
      }*/
      DistanceType dist = dataSource->distL2Squared(qid, id);

      result_set << Neighbor<IDType, DistanceType>(id, dist);

      return;
    }

    /* Which child branch should be taken first? */
    ElementType val = dataSource->get(qid, node->divfeat);
    DistanceType diff = val - node->divval;
    NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
    NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

    /* Create a branch record for the branch not taken.  Add distance
       of this feature boundary (we don't attempt to correct for any
       use of this feature in a parent node, which is unlikely to
       happen and would have only a small effect).  Don't bother
       adding more branches to heap after halfway point, as cost of
       adding exceeds their value.
       */

    DistanceType new_distsq = mindist + distance.accum_dist(val, node->divval, node->divfeat);

    /* Call recursively to search next level down. */
    searchLevelExact<with_removed>(qid, result_set, bestChild, mindist, epsError);

    if (mindist*epsError<=result_set.worstDist) {
      searchLevelExact<with_removed>(qid, result_set, otherChild, new_distsq, epsError);
    }
  }
  
  void ids_to_ids(const size_t* in, size_t* out, size_t size) const
  {
    // TODO
/*    if (removed_) {
      for (size_t i=0;i<size;++i) {
        out[i] = ids_[in[i]];
      }
    }*/
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
  enum 
  {
    SAMPLE_MEAN = 100,
    RAND_DIM = 5
  };
  const float updateCostWeight = 0.25; // lower -> more update

  size_t numTrees;
  Distance distance;
  size_t size = 0; // the number of points loaded into trees
  
  size_t sizeAtUpdate = 0;

  size_t dim;
  DataSource *dataSource;
  
  std::vector<IDType> ids;
  PooledAllocator pool;
  std::queue<NodeSplit> queue;

public:
  std::vector<KDTree<NodePtr>*> trees;
  UpdateStatus updateStatus = UpdateStatus::NoUpdate;
  KDTree<NodePtr>* ongoingTree;
  float queryLoss = 0.0;
};

}

#endif
