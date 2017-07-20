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

#include <progressive_base_index.h>

namespace panene
{

template <typename Distance, typename DataSource>
class KDTreeIndex : public ProgressiveBaseIndex<Distance, DataSource>
{
public:
  typedef typename DataSource::ElementType ElementType;
  typedef typename DataSource::DistanceType DistanceType;
  typedef typename DataSource::IDType IDType;

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
    
    NodeSplit(Node* node_, IDType *begin_, int count_) : node(node_), begin(begin_), count(count_) {}
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
  KDTreeIndex(int trees_, Distance distance_ = Distance()): trees(trees_), distance(distance_)
  {
  }

  KDTreeIndex(IndexParams indexParams_, Distance distance_ = Distance()): distance(distance_) {
    trees = indexParams_.trees;
  }

  ~KDTreeIndex() {
  }

  void setDataSource(DataSource *dataSource_) {
    dataSource = dataSource_;
    dim = dataSource -> dim();
  }

  size_t addPoints(size_t end) {
    size_t oldSize = size;
    size = end;

    if(size > dataSource -> loaded())
      size = dataSource -> loaded();
    
    if(sizeAtBuild * 2 < size) {
      buildIndex();
    }
    else {
      for(size_t i = oldSize; i < size; ++i) {
        for(int j = 0; j < trees; ++j) {
          addPointToTree(treeRoots[j], i);
        }
      }
    }
    return end - oldSize;
  }

  size_t update(int ops) {
    throw;
  }

  size_t getSize() { return size; }

  int usedMemory() const {
    return int(pool.usedMemory+pool.wastedMemory+size*sizeof(int));  // pool memory and vind array memory
  }

  void knnSearch(
      const IDType &qid,
      ResultSet<IDType, DistanceType> &resultSet,
      size_t knn,
      const SearchParams& params) const
  {
    bool use_heap = false; // TODO

    /*if (params.use_heap==FLANN_Undefined) {
      use_heap = (knn>KNN_HEAP_THRESHOLD)?true:false;
    }
    else {
      use_heap = (params.use_heap==FLANN_True)?true:false;
    }*/

    if (use_heap) {
      findNeighbors(qid, resultSet, params);
      //ids_to_ids(ids[i], ids[i], n);
    }
    else {
      findNeighbors(qid, resultSet, params);
      //ids_to_ids(ids[i], ids[i], n);
    }
  }

  void knnSearch(
      const std::vector<IDType> qids,
      std::vector<ResultSet<IDType, DistanceType>> &resultSets,
      size_t knn,
      const SearchParams& params) const
  {
    bool use_heap = false; // TODO

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
        for (int i = 0; i < (int)qids.size(); i++) {
          findNeighbors(qids[i], resultSets[i], params);
          //ids_to_ids(ids[i], ids[i], n);
        }
      }
    }
    else {
#pragma omp parallel num_threads(params.cores)
      {
#pragma omp for schedule(static)
        for (int i = 0; i < (int)qids.size(); i++) {
          findNeighbors(qids[i], resultSets[i], params);
          //ids_to_ids(ids[i], ids[i], n);
        }
      }
    }
  }

  void knnSearch(
      const std::vector<std::vector<ElementType>> &vectors,
      std::vector<ResultSet<IDType, DistanceType>> &resultSets,
      size_t knn,
      const SearchParams& params) const
  {
    bool use_heap = false; // TODO

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
    sizeAtBuild = size;
    std::vector<IDType> ids(size);
    for(size_t i = 0; i < size; ++i) {
      ids[i] = IDType(i);
    }

    treeRoots.resize(trees);

    for(int i = 0; i < trees; ++i) {
      std::random_shuffle(ids.begin(), ids.end());
      treeRoots[i] = divideTree(&ids[0], int(size));
    }
  }

  void freeIndex() {
    for(size_t i=0; i < treeRoots.size(); ++i) {
      if(treeRoots[i] != nullptr) treeRoots[i]->~Node();
    }
    pool.free();
  }
  
  void addPointToTree(NodePtr node, IDType id) {
//    ElementType* point = dataSource->get(id); // TODO

    if ((node->child1==NULL) && (node->child2==NULL)) {
//      ElementType *leafPoint = node->point;

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
    }
    else {
      if (dataSource->get(id, node->divfeat) < node->divval) {
          addPointToTree(node->child1, id);
      }
      else {
          addPointToTree(node->child2, id);                
      }
    }
  }

  NodePtr divideTree(IDType *ids, int count) {
    NodePtr node = new(pool) Node();
    
    /* If too few exemplars remain, then make this a leaf node. */
    if (count == 1) {
        node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
        node->divfeat = -1; // a leaf node
        node->id = ids[0];
    }
    else {
        int idx;
        int cutfeat;
        DistanceType cutval;
        meanSplit(ids, count, idx, cutfeat, cutval);

        node->divfeat = cutfeat;
        node->divval = cutval;
        node->child1 = divideTree(ids, idx);
        node->child2 = divideTree(ids+idx, count-idx);
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
    if (trees > 1) {
      fprintf(stderr,"It doesn't make any sense to use more than one tree for exact search");
    }
    if (trees > 0) {
      searchLevelExact<with_removed>(qid, result, treeRoots[0], 0.0, epsError);
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
    int i;
    BranchSt branch;

    int checkCount = 0;
    Heap<BranchSt>* heap = new Heap<BranchSt>((int)size);
    DynamicBitset checked(size);

    /* Search once through each tree down to root. */
    for (i = 0; i < trees; ++i) {
      searchLevel<with_removed>(qid, result, treeRoots[i], 0, checkCount, maxCheck, epsError, heap, checked);
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
    for (i = 0; i < trees; ++i) {
      searchLevel<with_removed>(vec, result, treeRoots[i], 0, checkCount, maxCheck, epsError, heap, checked);
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
      /*  Do not check same node more than once when searching multiple trees. */
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
      /*  Do not check same node more than once when searching multiple trees. */
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

private:
  enum 
  {
    SAMPLE_MEAN = 100,
    RAND_DIM = 5
  };

  int trees;
  Distance distance;
  size_t size = 0;
  size_t sizeAtUpdate = 0;
  size_t sizeAtBuild = 0;
  size_t dim;
  DataSource *dataSource;

  std::vector<NodePtr> treeRoots;
  NodePtr ongoingTree;
  std::vector<IDType> ids;
  PooledAllocator pool;
  std::queue<NodeSplit> queue;
  size_t replaced = 0;
};

}
#endif
