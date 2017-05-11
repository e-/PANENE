#ifndef panene_knn_table_h
#define panene_knn_table_h

#include <vector>
#include <iostream>
#include <queue>
#include <cassert>

#include "indices/kd_tree_index.h"
#include "indices/progressive_kd_tree_index.h"

using namespace panene;

#define DEBUG 0

template <class Indexer>
class KNNTable {
  typedef typename Indexer::ElementType ElementType;
  typedef typename Indexer::DistanceType DistanceType;
  typedef size_t IDType;
  
public:
  struct Neighbor {
    IDType id;    
    DistanceType distance;

    Neighbor() = default;
    Neighbor(IDType id_, DistanceType distance_) : id(id_), distance(distance_) {}

    friend std::ostream& operator<<( std::ostream& os, const Neighbor& obj ) {
      os << "(" << obj.id << ", " << obj.distance << ")";
      return os;  
    }

    bool operator< (const Neighbor &n) const {
      return this->distance < n.distance;
    }

    bool operator> (const Neighbor &n) const {
      return this->distance > n.distance;
    }
  };

  KNNTable(unsigned int k_, unsigned int d_, IndexParams indexParams_, SearchParams searchParams_, int maxOps_ = -1) : 
    d(d_), k(k_), indexer(Indexer(IndexParams(4))), maxOps(maxOps_) {

    indexer = Indexer(indexParams_);
    searchParams = searchParams_;
  }

  void setDataSource(Matrix<ElementType> &dataSource_) {
    dataSource = dataSource_;
    indexer.setDataSource(dataSource);
  }

  void addPoints(size_t end) {
    // We always add all points to the dataset regardless of the maxOps value to avoid losing data.
//    extendDataset(newPoints); 

    // calculate the number of steps allocated for each operation
    if(maxOps == -1) {
      maxOps = newPointOps = queueOps = indexOps = -1;
    }
    else {
      // example schedule
      newPointOps = maxOps / 4;
      queueOps = maxOps - newPointOps;
      indexOps = 0;
    }
  

    // 1. index new points (indexOps)
    
    size_t oldSize = indexer.getSize();

    // TODO: Add and update points to the index progressively
    // TODO: We do not have to insert ALL new points to the index. Rather, insert as many points as newPointOps

#if DEBUG
    std::cout << "adding points to indexer" << std::endl;
#endif

    indexer.addPoints(end);

    size_t size = indexer.getSize();
    size_t inc = size - oldSize;

#if DEBUG
    std::cout << inc << " new points have been indexed. " << indexer.getSize() << " points exist in trees." << std::endl;
#endif

    assert(indexer.getSize() > k); // check if at least k points are in the index

    Matrix<ElementType> newPoints(dataSource[oldSize], inc, dataSource.cols);

    // 2. compute new knn (newPointOps)
    // TODO: We need to keep a pointer, lastInserted and add dataset[lastInserted ... lastInserted + newPointOps]
  
    Matrix<IDType> indices(new IDType[newPoints.rows * k], newPoints.rows, k);
    Matrix<DistanceType> dists(new DistanceType[newPoints.rows * k], newPoints.rows, k);
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> queue; // descending order
    DynamicBitset checked(size);

#if DEBUG
    std::cerr << "knnSearch begins" << std::endl;
#endif 

    indexer.knnSearch(newPoints, indices, dists, k, searchParams);

#if DEBUG
    std::cerr << "knnSearch done" << std::endl;
#endif 

    std::vector<Neighbor> nns;
    nns.resize(k);

    for(unsigned int i = 0; i < newPoints.rows; ++i) {
      IDType id = oldSize + i;
      checked.set(id);

      for(unsigned int j = 0; j < k; ++j) {
        nns[j] = Neighbor(indices[i][j], dists[i][j]);
        queue.push(nns[j]);
      }
      
      neighbors.push_back(nns);
    }

    delete[] indices.ptr();
    delete[] dists.ptr();

#if DEBUG
    for(unsigned int i = 0; i < neighbors.size(); ++i) {
      for(unsigned int j = 0; j < k; ++j) {
        std::cout << neighbors[i][j] << ' ';
      }
      std::cout << std::endl;
    }
    std::cerr << "starting processing queue" << std::endl;
#endif

    // 3. Process the queue (queueOps)

    int checkCount = 0;
    Matrix<IDType> newIndices(new IDType[k], 1, k);
    Matrix<DistanceType> newDists(new DistanceType[k], 1, k);
    
    while(!queue.empty()) {
      auto q = queue.top();

#if DEBUG
      std::cerr << "Got a point from the queue: " << q << std::endl;
#endif
      queue.pop();

      if(checked.test(q.id)) continue;
      checked.set(q.id);
      checkCount++;

      // we need to update the NN of q.id
      
      // get new NN
      
      Matrix<ElementType> qvec(dataSource[q.id], 1, d);
#if DEBUG
      std::cerr << "starting knn search for queue" << std::endl;
#endif
      indexer.knnSearch(qvec, newIndices, newDists, k, searchParams);
#if DEBUG
      std::cerr << "done knn search for queue" << std::endl;
#endif

      
      // check if there is a difference between previous NN and newly computed NN.
      
      unsigned int i;
      for(i = 0; i < k; ++i) {
        if(neighbors[q.id][i].id != newIndices[0][i])
          break;
      } 

      if(i < k) { // if there is a difference
        // update q.id
        for(i = 0; i < k; ++i) {
          Neighbor ne(newIndices[0][i], newDists[0][i]);
          neighbors[q.id][i] = ne;
          if(!checked.test(ne.id))
            queue.push(ne);
        }
      }

#if DEBUG  
      std::cout << q << std::endl;
      for(auto &x : neighbors[q.id]) {
        std::cout << x << ' ';
      }
      std::cout << std::endl;

      for(unsigned int i = 0; i < k; ++i) {
        std::cout << '(' << newIndices[0][i] << ", " << newDists[0][i] << ") ";
      }
      std::cout << std::endl;
#endif

      if(queueOps >= 0 && checkCount >= queueOps) break;
    }

    delete[] newIndices.ptr();
    delete[] newDists.ptr();

    std::cout << checkCount << '\t';

#if DEBUG
    std::cout << checkCount << " points have been updated." << std::endl;
#endif
  };

  std::vector<Neighbor>& getNeighbors(const IDType id) {
    return neighbors[id];
  }

  int getMaxOps() { return maxOps; }
  void setMaxOps(int maxOps_) { maxOps = maxOps_; }

private:
  unsigned int d;
  unsigned int k;
  Indexer indexer;
  SearchParams searchParams;
  std::vector<std::vector<Neighbor>> neighbors;
  Matrix<ElementType> dataSource;

  /*
  maxOps indicates the maximum number of operations that the function addPoints can execute.
  There are three types of operations:
    1) Add a new point P to the table. It requires:
      An insertion operation to the index
      A knn search for P
      Mark the neighbors of P as dirty and insert them to the queue
    
    2) Take a dirty point P from the queue and update its neighbors
      A knn search for P
      Mark the neighbors of P as dirty and insert them to the queue

    3) Rebalance the index
      Update only one tree at a time

  Note that the three operations have different costs. 
  */
  
  int maxOps = -1; // unlimited 
  int newPointOps = -1; // add new rows to the table
  int queueOps = -1; // take a point from the queue and update its neighbors
  int indexOps = -1; // update index 
};

#endif
