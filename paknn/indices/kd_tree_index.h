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

  KDTreeIndex(int trees_): trees(trees_)
  {

  }

  void setDataSource(Matrix<ElementType> &dataSource_) {
    dataSource = dataSource_;
  }

  void addPoints() {

  }

  int usedMemory() const {

    return 1;
  }

  void findNeighbors() {

  }

protected:
  
  void buildIndex() {

  }

  void freeIndex() {

  }

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

  NodePtr divideTree() {
    return nullptr;
  }

  void meanSplit() {
  }
  
  int selectDivision() {
    return 0;
  }

  void planeSplit() {
  }
  
  void getExactNeighbors() {

  }

  void getNeighbors() {

  }

  void searchLevel() {

  }
  
  void searchLevelExact() {

  }

  void addPointToTree() {

  }

private:
  int trees;
  Matrix<ElementType> dataSource;

  DistanceType* mean;
  DistanceType* var;
  std::vector<NodePtr> treeRoots;
  PooledAllocator pool;
};

}
#endif
