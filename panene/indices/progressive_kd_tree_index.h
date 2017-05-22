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

#include "progressive_base_index.h"
#include "../data/data_source.h"

namespace panene
{

template <typename Distance>
class ProgressiveKDTreeIndex : public ProgressiveBaseIndex<Distance>
{
public:
  typedef DataSource::ElementType ElementType;
  typedef DataSource::DistanceType DistanceType;


protected:
  struct Node {
    int divfeat;
    DistanceType divval;
    ElementType* point;
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
    struct Node* node;
    int *begin;
    int count;
    
    NodeSplit(Node* node_, int* begin_, int count_) : node(node_), begin(begin_), count(count_) {}
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
  ProgressiveKDTreeIndex(int trees_, Distance distance_ = Distance()): trees(trees_), distance(distance_)
  {
  }

  ProgressiveKDTreeIndex(IndexParams indexParams_, Distance distance_ = Distance()): distance(distance_) {
    trees = indexParams_.trees;
  }

  ~ProgressiveKDTreeIndex() {
    if(mean != nullptr) delete[] mean;
    if(var != nullptr)delete[] var;
  }

  void setDataSource(DataSource *dataSource_) {
    dataSource = dataSource_;
    veclen = dataSource -> dim();

    mean = new DistanceType[veclen];
    var = new DistanceType[veclen];
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
      for(size_t i=oldSize; i<size; ++i) {
        for(int j = 0; j< trees; ++j) {
          addPointToTree(treeRoots[j], i);
        }
      }
      return size - oldSize;
    }
  }

  size_t update(int ops) {
    if(sizeAtUpdate == 0) {
      sizeAtUpdate = size;
      indices.resize(sizeAtUpdate);

      for(size_t i = 0; i < sizeAtUpdate; ++i) indices[i] = int(i);
      std::random_shuffle(indices.begin(), indices.end());
      
      ongoingTree = new(pool) Node();
      queue.push(NodeSplit(ongoingTree, &indices[0], sizeAtUpdate));
    }
    
    int updatedCount = 0;

    while((ops == -1 || updatedCount < ops) && !queue.empty()) {
      NodeSplit nodeSplit = queue.front();
      queue.pop(); 
      
//      std::cerr << "updatedCount " << updatedCount << std::endl;

      NodePtr node = nodeSplit.node;
      int *begin = nodeSplit.begin;
      int count = nodeSplit.count;

//      std::cerr << begin << " " << count << std::endl;

      // At this point, nodeSplit the two children of nodeSplit are nullptr
      if (count == 1) {
          node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
          node->divfeat = *begin;    /* Store index of this vec. */
          node->point = dataSource->get(*begin);
      }
      else {
          int idx;
          int cutfeat;
          DistanceType cutval;
          meanSplit(begin, count, idx, cutfeat, cutval);
          
//          std::cerr << "cut index: " << idx << " cut count: " << count << std::endl;

          node->divfeat = cutfeat;
          node->divval = cutval;
          node->child1 = new(pool) Node();
          node->child2 = new(pool) Node();
          
          queue.push(NodeSplit(node->child1, begin, idx));
          queue.push(NodeSplit(node->child2, begin + idx, count - idx));
      }
      updatedCount++;
    }

    if(queue.empty()) { //finished creating a new tree
      // reset the sizeAtUpdate
      sizeAtUpdate = 0;
      
      // get the victim
      NodePtr victim = treeRoots[replaced % trees];
      
      // replace the victim with the newly created tree
      treeRoots[replaced % trees]  = ongoingTree;
      replaced++;

      // free the victim
      victim -> ~Node();
    }

    return updatedCount;
  }

  size_t getSize() { return size; }

  int usedMemory() const {
  	return int(pool.usedMemory+pool.wastedMemory+size*sizeof(int));  // pool memory and vind array memory
  }

	/**
	 * Find set of nearest neighbors to vec. Their indices are stored inside
	 * the result object.
	 *
	 * Params:
	 *     result = the result object in which the indices of the nearest-neighbors are stored
	 *     vec = the vector for which to search the nearest neighbors
	 *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
	 */
	void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
	{
		int maxChecks = searchParams.checks;
		float epsError = 1+searchParams.eps;

		if (maxChecks==0) { // }FLANN_CHECKS_UNLIMITED) {
			getExactNeighbors<false>(result, vec, epsError);
			/*if (removed_) {
				getExactNeighbors<true>(result, vec, epsError);
			}
			else {
				getExactNeighbors<false>(result, vec, epsError);
			}*/
		}
		else {
			getNeighbors<false>(result, vec, maxChecks, epsError);
			/*if (removed_) {
				getNeighbors<true>(result, vec, maxChecks, epsError);
			}
			else {
				getNeighbors<false>(result, vec, maxChecks, epsError);
			}*/
		}
	}


	/**
	 * @brief Perform k-nearest neighbor search
	 * @param[in] queries The query points for which to find the nearest neighbors
	 * @param[out] indices The indices of the nearest neighbors found
	 * @param[out] dists Distances to the nearest neighbors found
	 * @param[in] knn Number of nearest neighbors to return
	 * @param[in] params Search parameters
	 */
  int knnSearch(const Matrix<ElementType>& queries,
			Matrix<size_t>& indices,
			Matrix<DistanceType>& dists,
			size_t knn,
			const SearchParams& params) const
	{
		assert(queries.cols == veclen);
		assert(indices.rows >= queries.rows);
		assert(dists.rows >= queries.rows);
		assert(indices.cols >= knn);
		assert(dists.cols >= knn);
		bool use_heap = false; // TODO

		/*if (params.use_heap==FLANN_Undefined) {
			use_heap = (knn>KNN_HEAP_THRESHOLD)?true:false;
		}
		else {
			use_heap = (params.use_heap==FLANN_True)?true:false;
		}*/
		int count = 0;

		if (use_heap) {
#pragma omp parallel num_threads(params.cores)
			{
				KNNResultSet2<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					findNeighbors(resultSet, queries[i], params);
					size_t n = std::min(resultSet.size(), knn);
					resultSet.copy(indices[i], dists[i], n, params.sorted);
					indices_to_ids(indices[i], indices[i], n);
					count += n;
				}
			}
		}
		else {
#pragma omp parallel num_threads(params.cores)
			{
				KNNSimpleResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					findNeighbors(resultSet, queries[i], params);
					size_t n = std::min(resultSet.size(), knn);

					resultSet.copy(indices[i], dists[i], n, params.sorted);

					indices_to_ids(indices[i], indices[i], n);
					count += n;
				}
			}
		}
		return count;
	}

protected:
  
  void buildIndex() {
    std::vector<int> ind(size);
    for(size_t i = 0; i < size; ++i) {
      ind[i] = int(i);
    }

    treeRoots.resize(trees);

    for(int i = 0; i < trees; ++i) {
      std::random_shuffle(ind.begin(), ind.end());
      //std::cerr << "first five indices: " << ind[0] << " " << ind[1] << " " << ind[2] << " " << ind[3] << " " <<           ind[4] << std::endl;

      treeRoots[i] = divideTree(&ind[0], int(size));
    }
  }

  void freeIndex() {
    for(size_t i=0; i < treeRoots.size(); ++i) {
      if(treeRoots[i] != nullptr) treeRoots[i]->~Node();
    }
    pool.free();
  }
  
  void addPointToTree(NodePtr node, int ind) {
		ElementType* point = dataSource->get(ind);

		if ((node->child1==NULL) && (node->child2==NULL)) {
      ElementType* leafPoint = node->point;
      ElementType maxSpan = 0;

      size_t divfeat = 0;
      for (size_t i=0;i<veclen;++i) {
          ElementType span = std::abs(point[i]-leafPoint[i]);
          if (span > maxSpan) {
              maxSpan = span;
              divfeat = i;
          }
      }

      NodePtr left = new(pool) Node();
      left->child1 = left->child2 = NULL;

      NodePtr right = new(pool) Node();
      right->child1 = right->child2 = NULL;

      if (point[divfeat]<leafPoint[divfeat]) {
          left->divfeat = ind;
          left->point = point;
          right->divfeat = node->divfeat;
          right->point = node->point;
      }
      else {
          left->divfeat = node->divfeat;
          left->point = node->point;
          right->divfeat = ind;
          right->point = point;
      }
      node->divfeat = divfeat;
      node->divval = (point[divfeat]+leafPoint[divfeat])/2;
      node->child1 = left;
      node->child2 = right;            
		}
		else {
      if (point[node->divfeat]<node->divval) {
          addPointToTree(node->child1,ind);
      }
      else {
          addPointToTree(node->child2,ind);                
      }
		}
  }

  NodePtr divideTree(int *ind, int count) {
		NodePtr node = new(pool) Node();
		
    /* If too few exemplars remain, then make this a leaf node. */
		if (count == 1) {
				node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
				node->divfeat = *ind;    /* Store index of this vec. */
				node->point = dataSource -> get(*ind);
		}
		else {
				int idx;
				int cutfeat;
				DistanceType cutval;
				meanSplit(ind, count, idx, cutfeat, cutval);
//        std::cerr << "cutfeat: " << cutfeat << " cutval: " << cutval << " count: " << count << std::endl;

				node->divfeat = cutfeat;
				node->divval = cutval;
				node->child1 = divideTree(ind, idx);
				node->child2 = divideTree(ind+idx, count-idx);
		}

		return node;
  }

   /**
    * Choose which feature to use in order to subdivide this set of vectors.
    * Make a random choice among those with the highest variance, and use
    * its variance as the threshold value.
    **/
  void meanSplit(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval) {
		memset(mean,0,veclen*sizeof(DistanceType));
		memset(var,0,veclen*sizeof(DistanceType));

		/* Compute mean values.  Only the first SAMPLE_MEAN values need to be
				sampled to get a good estimate.
		 */
		int cnt = std::min((int)SAMPLE_MEAN+1, count);
		for (int j = 0; j < cnt; ++j) {
				ElementType* v = dataSource -> get(ind[j]);
				for (size_t k=0; k<veclen; ++k) {
						mean[k] += v[k];
				}
		}
		DistanceType divFactor = DistanceType(1)/cnt;
		for (size_t k=0; k<veclen; ++k) {
				mean[k] *= divFactor;
		}

		/* Compute variances (no need to divide by count). */
		for (int j = 0; j < cnt; ++j) {
				ElementType* v = dataSource -> get(ind[j]);
				for (size_t k=0; k<veclen; ++k) {
						DistanceType dist = v[k] - mean[k];
						var[k] += dist * dist;
				}
		}
		/* Select one of the highest variance indices at random. */
		cutfeat = selectDivision(var);
		cutval = mean[cutfeat];

		int lim1, lim2;
		planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

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
	int selectDivision(DistanceType* v)
	{
		int num = 0;
		size_t topind[RAND_DIM];

		/* Create a list of the indices of the top RAND_DIM values. */
		for (size_t i = 0; i < veclen; ++i) {
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
	void planeSplit(int* ind, int count, int cutfeat, DistanceType cutval, int& lim1, int& lim2)
	{
		/* Move vector indices for left subtree to front of list. */
		int left = 0;
		int right = count-1;
		for (;; ) {
			while (left<=right && dataSource -> get(ind[left])[cutfeat]<cutval) ++left;
			while (left<=right && dataSource -> get(ind[right])[cutfeat]>=cutval) --right;
			if (left>right) break;
			std::swap(ind[left], ind[right]); ++left; --right;
		}
		lim1 = left;
		right = count-1;
		for (;; ) {
			while (left<=right && dataSource -> get(ind[left])[cutfeat]<=cutval) ++left;
			while (left<=right && dataSource -> get(ind[right])[cutfeat]>cutval) --right;
			if (left>right) break;
			std::swap(ind[left], ind[right]); ++left; --right;
		}
		lim2 = left;
	}

	/**
	 * Performs an exact nearest neighbor search. The exact search performs a full
	 * traversal of the tree.
	 */
	template<bool with_removed>
	void getExactNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, float epsError) const
	{
		if (trees > 1) {
			fprintf(stderr,"It doesn't make any sense to use more than one tree for exact search");
		}
		if (trees > 0) {
			searchLevelExact<with_removed>(result, vec, treeRoots[0], 0.0, epsError);
		}
	}

	/**
	 * Performs the approximate nearest-neighbor search. The search is approximate
	 * because the tree traversal is abandoned after a given number of descends in
	 * the tree.
	 */
	template<bool with_removed>
	void getNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, int maxCheck, float epsError) const
	{
		int i;
		BranchSt branch;

		int checkCount = 0;
		Heap<BranchSt>* heap = new Heap<BranchSt>((int)size);
		DynamicBitset checked(size);

		/* Search once through each tree down to root. */
		for (i = 0; i < trees; ++i) {
			searchLevel<with_removed>(result, vec, treeRoots[i], 0, checkCount, maxCheck, epsError, heap, checked);
		}

		/* Keep searching other branches from heap until finished. */
		while ( heap->popMin(branch) && (checkCount < maxCheck || !result.full() )) {
			searchLevel<with_removed>(result, vec, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
		}

		delete heap;

	}

	/**
	 *  Search starting from a given node of the tree.  Based on any mismatches at
	 *  higher levels, all exemplars below this level must have a distance of
	 *  at least "mindistsq".
	 */
	template<bool with_removed>
	void searchLevel(ResultSet<DistanceType>& result_set, const ElementType* vec, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
			float epsError, Heap<BranchSt>* heap, DynamicBitset& checked) const
	{
		if (result_set.worstDist()<mindist) {
			//			printf("Ignoring branch, too far\n");
			return;
		}

		/* If this is a leaf node, then do check and return. */
		if ((node->child1 == NULL)&&(node->child2 == NULL)) {
			int index = node->divfeat;
      // TODO
/*			if (with_removed) {
				if (removed_points_.test(index)) return;
			}*/
			/*  Do not check same node more than once when searching multiple trees. */
			if ( checked.test(index) || ((checkCount>=maxCheck)&& result_set.full()) ) return;
			checked.set(index);
			checkCount++;

			DistanceType dist = distance(node->point, vec, veclen);
			result_set.addPoint(dist,index);
			return;
		}

		/* Which child branch should be taken first? */
		ElementType val = vec[node->divfeat];
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
		//		if (2 * checkCount < maxCheck  ||  !result.full()) {
		if ((new_distsq*epsError < result_set.worstDist())||  !result_set.full()) {
			heap->insert( BranchSt(otherChild, new_distsq) );
		}

		/* Call recursively to search next level down. */
		searchLevel<with_removed>(result_set, vec, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
	}

	/**
	 * Performs an exact search in the tree starting from a node.
	 */
	template<bool with_removed>
	void searchLevelExact(ResultSet<DistanceType>& result_set, const ElementType* vec, const NodePtr node, DistanceType mindist, const float epsError) const
	{
		/* If this is a leaf node, then do check and return. */
		if ((node->child1 == NULL)&&(node->child2 == NULL)) {
			int index = node->divfeat;
      // TODO
/*			if (with_removed) {
				if (removed_points_.test(index)) return; // ignore removed points
			}*/
			DistanceType dist = distance(node->point, vec, veclen);
			result_set.addPoint(dist,index);

			return;
		}

		/* Which child branch should be taken first? */
		ElementType val = vec[node->divfeat];
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
		searchLevelExact<with_removed>(result_set, vec, bestChild, mindist, epsError);

		if (mindist*epsError<=result_set.worstDist()) {
			searchLevelExact<with_removed>(result_set, vec, otherChild, new_distsq, epsError);
		}
	}
  
  void indices_to_ids(const size_t* in, size_t* out, size_t size) const
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
  size_t veclen;
  DataSource *dataSource;

  DistanceType* mean = nullptr;
  DistanceType* var = nullptr;
  std::vector<NodePtr> treeRoots;
  NodePtr ongoingTree;
  std::vector<int> indices;
  PooledAllocator pool;
  std::queue<NodeSplit> queue;
  size_t replaced = 0;
};

}
#endif
