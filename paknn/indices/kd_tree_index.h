#ifndef kd_tree_index_h
#define kd_tree_index_h

#include <vector>

#include "../util/matrix.h"
#include "../util/allocator.h"

namespace paknn
{

// TODO use pool
//

class KDTreeIndex
{
public:
  typedef float ElementType;
  typedef float DistanceType;

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
  KDTreeIndex(int trees_): trees(trees_)
  {

  }

  void setDataSource(Matrix<ElementType> &dataSource_) {
    dataSource = dataSource_;
    veclen = dataSource.cols;
  }

  void addPoints(int begin, size_t end) {
    size_t oldSize = size;
    size = end;

    if(sizeAtBuild * 2 < size) {
      buildIndex();
    }
    else {
      for(size_t i=oldSize; i<size; ++i) {
        for(int j = 0; j< trees; ++j) {
          addPointToTree(treeRoots[j], i);
        }
      }
    }
  }

  int usedMemory() const {
  	return int(pool.usedMemory+pool.wastedMemory+size*sizeof(int));  // pool memory and vind array memory
  }

  void findNeighbors() {

  }

protected:
  
  void buildIndex() {
    sizeAtBuild = size;
    std::vector<int> ind(size);
    for(size_t i = 0; i < treeRoots.size(); ++i) {
      ind[i] = int(i);
    }

    mean = new DistanceType[veclen];
    var = new DistanceType[veclen];

    treeRoots.resize(trees);

    for(int i = 0; i < trees; ++i) {
      std::random_shuffle(ind.begin(), ind.end());
      treeRoots[i] = divideTree(&ind[0], int(size));
    }

    delete[] mean;
    delete[] var;
  }

  void freeIndex() {
    for(size_t i=0; i < treeRoots.size(); ++i) {
      if(treeRoots[i] != nullptr) treeRoots[i]->~Node();
    }
    pool.free();
  }
  
  void addPointToTree(NodePtr node, int ind) {
		ElementType* point = dataSource[ind];

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
				node->point = dataSource[*ind];
		}
		else {
				int idx;
				int cutfeat;
				DistanceType cutval;
				meanSplit(ind, count, idx, cutfeat, cutval);

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
				ElementType* v = dataSource[ind[j]];
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
				ElementType* v = dataSource[ind[j]];
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
			while (left<=right && dataSource[ind[left]][cutfeat]<cutval) ++left;
			while (left<=right && dataSource[ind[right]][cutfeat]>=cutval) --right;
			if (left>right) break;
			std::swap(ind[left], ind[right]); ++left; --right;
		}
		lim1 = left;
		right = count-1;
		for (;; ) {
			while (left<=right && dataSource[ind[left]][cutfeat]<=cutval) ++left;
			while (left<=right && dataSource[ind[right]][cutfeat]>cutval) --right;
			if (left>right) break;
			std::swap(ind[left], ind[right]); ++left; --right;
		}
		lim2 = left;
	}
  
  void getExactNeighbors() {

  }

  void getNeighbors() {

  }

  void searchLevel() {

  }
  
  void searchLevelExact() {

  }

private:
  enum 
  {
    SAMPLE_MEAN = 100,
    RAND_DIM = 5
  };

  int trees;
  size_t size;
  size_t sizeAtBuild;
  size_t veclen;
  Matrix<ElementType> dataSource;

  DistanceType* mean;
  DistanceType* var;
  std::vector<NodePtr> treeRoots;
  PooledAllocator pool;
};

}
#endif
