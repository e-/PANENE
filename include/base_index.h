#ifndef panene_base_index_h
#define panene_base_index_h

#include <vector>
#include <algorithm>
#include <random>

#include <util/allocator.h>
#include <util/heap.h>
#include <util/dynamic_bitset.h>
#include <util/random.h>
#include <util/result_set.h>
#include <dist.h>

#include <kd_tree.h>

#include <roaring/roaring.hh>
#include <roaring/roaring.c>

#define USE_BASECLASS_SYMBOLS public: \
  typedef typename BaseIndex<DataSource>::Distance Distance;\
  typedef typename BaseIndex<DataSource>::IDType IDType;\
  typedef typename BaseIndex<DataSource>::DistanceType DistanceType;\
  typedef typename BaseIndex<DataSource>::ElementType ElementType;\
  typedef typename BaseIndex<DataSource>::Node Node;\
  typedef typename BaseIndex<DataSource>::NodePtr NodePtr;\
  typedef typename BaseIndex<DataSource>::BranchSt BranchSt;\
  typedef typename BaseIndex<DataSource>::Branch Branch;\
  typedef typename BaseIndex<DataSource>::NodeSplit NodeSplit;\
  using BaseIndex<DataSource>::size;\
  using BaseIndex<DataSource>::dataSource;\
  using BaseIndex<DataSource>::numTrees;\
  using BaseIndex<DataSource>::trees;\
  using BaseIndex<DataSource>::pool;\
  using BaseIndex<DataSource>::dim;\
  using BaseIndex<DataSource>::meanSplit;\
  using BaseIndex<DataSource>::divideTree;\
  using BaseIndex<DataSource>::findNeighbors;

namespace panene
{

struct IndexParams {
  int trees;

  IndexParams(int trees_ = 4) : trees(trees_) {}
};

struct SearchParams {
  int checks; // 32
  float eps; // 0
  int sorted;
  int cores;
  Roaring *mask = nullptr;

  SearchParams(int checks_ = 32, float eps_ = 0, int sorted_ = 0, int cores_ = 0) : checks(checks_), eps(eps_), sorted(sorted_), cores(cores_) {}
};

template <typename DataSource>
class BaseIndex
{
public:
  typedef typename DataSource::Distance Distance;
  typedef typename DataSource::IDType IDType;
  typedef typename DataSource::ElementType ElementType;
  typedef typename DataSource::DistanceType DistanceType;

public:
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
      if (child1 != nullptr) { child1->~Node(); child1 = nullptr; }
      if (child2 != nullptr) { child2->~Node(); child2 = nullptr; }
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

  BaseIndex(DataSource *dataSource_, IndexParams indexParams_, Distance distance_ = Distance()) : dataSource(dataSource_), distance(distance_) {
    numTrees = indexParams_.trees;
    trees.resize(numTrees);
    dim = dataSource->dim();
    for (size_t i = 0; i < numTrees; ++i) {
      trees[i] = new KDTree<NodePtr>(dataSource->capacity());
    }
  }

  ~BaseIndex() {
    for (size_t i = 0; i < numTrees; ++i) {
      delete trees[i];
    }
  }

  virtual size_t addPoints(size_t end) = 0;

  virtual size_t update(int ops) = 0;

  void removePoint(size_t id) {
    if (!removed) {
      removedPoints.resize(dataSource->size());
      removedPoints.reset();
      removed = true;
    }

    if (!removedPoints.test(id)) {
      removedPoints.set(id);
      removedCount++;
    }
  }

  size_t getSize() { return size; }

  int usedMemory() const {
    return int(pool.usedMemory + pool.wastedMemory + size * sizeof(int));  // pool memory and vind array memory
  }

  /*virtual void knnSearch(
      const IDType &qid,
      ResultSet<IDType, DistanceType> &resultSets,
      size_t knn,
      const SearchParams& params) const = 0;
  
  virtual void knnSearch(
    const std::vector<std::vector<ElementType>> &vectors,
      std::vector<ResultSet<IDType, DistanceType>> &resultSets,
      size_t knn,
      const SearchParams& params) const = 0;*/


  /**
  * Find set of nearest neighbors to vec. Their ids are stored inside
  * the result object.
  *
  * Params:
  *     result = the result object in which the ids of the nearest-neighbors are stored
  *     vec = the vector for which to search the nearest neighbors
  *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
  */

  void findNeighbors(const std::vector<ElementType> &vec, ResultSet<IDType, DistanceType> &result, const SearchParams& searchParams) const
  {
    int maxChecks = searchParams.checks;
    float epsError = 1 + searchParams.eps;
    Roaring *mask = searchParams.mask;

    if (removed) {
      getNeighbors<true>(vec, result, maxChecks, epsError, mask);
    }
    else {
      getNeighbors<false>(vec, result, maxChecks, epsError, mask);
    }

    // TODO
    for(size_t i = 0; i < result.k; ++i) 
      result.distances[i] = sqrt(result.distances[i]);
  }

  /**
  * Performs the approximate nearest-neighbor search. The search is approximate
  * because the tree traversal is abandoned after a given number of descends in
  * the tree.
  */
  template<bool with_removed>
  void getNeighbors(const std::vector<ElementType> &vec, ResultSet<IDType, DistanceType> &result, int maxCheck, float epsError, Roaring *mask) const
  {
    BranchSt branch;

    int checkCount = 0;

    Heap<BranchSt>* heap = new Heap<BranchSt>((int)size);
    DynamicBitset checked(size);

    /* Search once through each tree down to root. */
    for (size_t i = 0; i < numTrees; ++i) {
      searchLevel<with_removed>(vec, result, trees[i]->root, 0, checkCount, maxCheck, epsError, heap, checked, mask);
    }

    /* Keep searching other branches from heap until finished. */
    while (heap->popMin(branch) && (checkCount < maxCheck || !result.full())) {
      searchLevel<with_removed>(vec, result, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked, mask);
    }

    delete heap;
  }

  /**
  *  Search starting from a given node of the tree.  Based on any mismatches at
  *  higher levels, all exemplars below this level must have a distance of
  *  at least "mindistsq".
  */
  template<bool with_removed>
  void searchLevel(const std::vector<ElementType> &vec, ResultSet<IDType, DistanceType> &result_set, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
    float epsError, Heap<BranchSt>* heap, DynamicBitset& checked, Roaring *mask) const
  {
    if (result_set.worstDist < mindist) {
      //      printf("Ignoring branch, too far\n");
      return;
    }

    /* If this is a leaf node, then do check and return. */
    if ((node->child1 == NULL) && (node->child2 == NULL)) {
      IDType id = node->id;

      if (with_removed && removedPoints.test(id)) return;

      if (mask != nullptr && mask->contains(id)) return;

      /*  Do not check same node more than once when searching multiple numTrees. */
      if (checked.test(id) || ((checkCount >= maxCheck) && result_set.full())) return;
      checked.set(id);
      checkCount++;

      DistanceType dist = dataSource->getSquaredDistance(id, vec);
      result_set << Neighbor<IDType, DistanceType>(id, dist);
      return;
    }

    /* Which child branch should be taken first? */
    ElementType val = vec[node->divfeat]; //dataSource->get(qid, node -> divfeat);
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
    if ((new_distsq*epsError < result_set.worstDist) || !result_set.full()) {
      heap->insert(BranchSt(otherChild, new_distsq));
    }

    /* Call recursively to search next level down. */
    searchLevel<with_removed>(vec, result_set, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked, mask);
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
      node->child2 = divideTree(tree, ids + idx, count - idx, depth + 1);
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

    int sampleCount = (std::min)((int)SAMPLE_MEAN + 1, count);
    std::vector<DistanceType> mean(dim), var(dim);

    dataSource->computeMeanAndVar(ids, sampleCount, mean, var);

    /* Select one of the highest variance ids at random. */
    cutfeat = selectDivision(var);
    cutval = mean[cutfeat];

    int lim1, lim2;
    planeSplit(ids, count, cutfeat, cutval, lim1, lim2);

    if (lim1>count / 2) index = lim1;
    else if (lim2<count / 2) index = lim2;
    else index = count / 2;

    /* If either list is empty, it means that all remaining features
    * are identical. Split in the middle to maintain a balanced tree.
    */
    if ((lim1 == count) || (lim2 == 0)) index = count / 2;
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
      if ((num < RAND_DIM) || (v[i] > v[topind[num - 1]])) {
        /* Put this element at end of topind. */
        if (num < RAND_DIM) {
          topind[num++] = i;            /* Add to list. */
        }
        else {
          topind[num - 1] = i;         /* Replace last element. */
        }
        /* Bubble end value down to right location by repeated swapping. */
        int j = num - 1;
        while (j > 0 && v[topind[j]] > v[topind[j - 1]]) {
          std::swap(topind[j], topind[j - 1]);
          --j;
        }
      }
    }
    /* Select a random integer in range [0,num-1], and return that index. */
    int rnd = rand_int(num);
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
    int right = count - 1;
    for (;; ) {
      while (left <= right && dataSource->get(ids[left], cutfeat) < cutval) ++left; // TODO
      while (left <= right && dataSource->get(ids[right], cutfeat) >= cutval) --right; // TODO
      if (left>right) break;
      std::swap(ids[left], ids[right]); ++left; --right;
    }
    lim1 = left;
    right = count - 1;
    for (;; ) {
      while (left <= right && dataSource->get(ids[left], cutfeat) <= cutval) ++left; // TODO
      while (left <= right && dataSource->get(ids[right], cutfeat) > cutval) --right; // TODO
      if (left>right) break;
      std::swap(ids[left], ids[right]); ++left; --right;
    }
    lim2 = left;
  }

  virtual void freeIndex() = 0;

public:
  enum
  {
    SAMPLE_MEAN = 100,
    RAND_DIM = 5
  };

  size_t numTrees;
  Distance distance;
  size_t size = 0; // the number of points loaded into trees
  size_t dim;
  DataSource *dataSource;

  bool removed = false;
  DynamicBitset removedPoints;
  size_t removedCount = 0;

  PooledAllocator pool;

public:
  std::vector<KDTree<NodePtr>*> trees;
};

}

#endif
