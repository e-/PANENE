#ifndef panene_progressive_knn_table_h
#define panene_progressive_knn_table_h

#include <vector>
#include <iostream>
#include <queue>
#include <cassert>

#include "indices/kd_tree_index.h"
#include "indices/progressive_kd_tree_index.h"
#include "scheduler/schedule.h"
#include "scheduler/naive_scheduler.h"

using namespace panene;

#define DEBUG 0

template <class Indexer>
class ProgressiveKNNTable {
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

  struct UpdateResult {
    Schedule schedule;
    size_t addNewPointResult;
    size_t updateIndexResult;
    size_t updateTableResult;

    UpdateResult(const Schedule &schedule_, size_t addNewPointResult_, size_t updateIndexResult_, size_t updateTableResult_) :
      schedule(schedule_), addNewPointResult(addNewPointResult_), updateIndexResult(updateIndexResult_), updateTableResult(updateTableResult_) { 
    }

    friend std::ostream& operator<<( std::ostream& os, const UpdateResult& obj ) {
      os << "UpdateResult(addNewPointOps: " << obj.addNewPointResult << " / " << obj.schedule.addNewPointOps << ", " 
         << "updateIndexOps: " << obj.updateIndexResult << " / " << obj.schedule.updateIndexOps << ", " 
         << "updateTableOps: " << obj.updateTableResult << " / " << obj.schedule.updateTableOps << ")";
      return os;  
    }
  };

  ProgressiveKNNTable(unsigned int k_, unsigned int d_, IndexParams indexParams_, SearchParams searchParams_) : 
    d(d_), k(k_), indexer(Indexer(IndexParams(4))) {
    
    indexer = Indexer(indexParams_);
    searchParams = searchParams_;
  }

  void setDataSource(Matrix<ElementType> &dataSource_) {
    dataSource = dataSource_;
    indexer.setDataSource(dataSource);
    checked = DynamicBitset(dataSource.rows);
  }

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
  UpdateResult update(size_t maxOps) {
    // To avoid keeping a copy of the whole data, we need to use an abstract dataframe that grows in real time.
    // Since we do not have such a dataframe currently, we assume all data are loaded in a datasource.
    
    // calculate the number of steps allocated for each operation
    Schedule schedule = naiveScheduler.schedule(maxOps);
#if DEBUG
    std::cerr << "update with schedule " << schedule << std::endl;
#endif

    size_t oldSize = indexer.getSize();

    // 1. index new points (addNewPointps)
    
#if DEBUG
    std::cerr << "adding points to indexer" << std::endl;
#endif

    size_t addNewPointResult = indexer.addPoints(schedule.addNewPointOps);

#if DEBUG
    std::cerr << addNewPointResult << " new points have been indexed. " << indexer.getSize() << " points exist in trees." << std::endl;
#endif

    // 2. update the index (updateIndexOps)
    
    size_t updateIndexResult = indexer.update(schedule.updateIndexOps);
    
    // 3. compute knn for newly added points in step 1 (newPointOps)
    
    size_t size = indexer.getSize();

    if(addNewPointResult > 0) {
      assert(size > k); // check if at least k points are in the index

      Matrix<ElementType> newPoints(dataSource[size - addNewPointResult], addNewPointResult, dataSource.cols);

      Matrix<IDType> indices(new IDType[newPoints.rows * k], newPoints.rows, k);
      Matrix<DistanceType> dists(new DistanceType[newPoints.rows * k], newPoints.rows, k);

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
#if DEBUG
          std::cerr << "queue added " << nns[j] << std::endl;
#endif
        }
        
        neighbors.push_back(nns);
      }

      delete[] indices.ptr();
      delete[] dists.ptr();
    }


#if DEBUG
    for(unsigned int i = 0; i < neighbors.size(); ++i) {
      for(unsigned int j = 0; j < k; ++j) {
        std::cerr << neighbors[i][j] << ' ';
      }
      std::cerr << std::endl;
    }
    std::cerr << "starting processing queue" << std::endl;
#endif

    // 4. process the queue (updateTableOps)

    int checkCount = 0;
    Matrix<IDType> newIndices(new IDType[k], 1, k);
    Matrix<DistanceType> newDists(new DistanceType[k], 1, k);
    
    while((schedule.updateTableOps < 0 || checkCount < schedule.updateTableOps) && !queue.empty()) {
      auto q = queue.top();

#if DEBUG
      std::cerr << "Got a point from the queue: " << q << std::endl;
#endif
      queue.pop();

      checked.reset(q.id);
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
      std::cerr << q << std::endl;
      for(auto &x : neighbors[q.id]) {
        std::cerr << x << ' ';
      }
      std::cerr << std::endl;

      for(unsigned int i = 0; i < k; ++i) {
        std::cerr << '(' << newIndices[0][i] << ", " << newDists[0][i] << ") ";
      }
      std::cerr << std::endl;
#endif
    }

    delete[] newIndices.ptr();
    delete[] newDists.ptr();

#if DEBUG
    std::cerr << checkCount << " points have been updated." << std::endl;
#endif
    
    size_t updateTableResult = checkCount;

    return UpdateResult(schedule, addNewPointResult, updateIndexResult, updateTableResult);
  };

  std::vector<Neighbor>& getNeighbors(const IDType id) {
    return neighbors[id];
  }

private:
  unsigned int d;
  unsigned int k;
  Indexer indexer;
  SearchParams searchParams;
  std::vector<std::vector<Neighbor>> neighbors;
  Matrix<ElementType> dataSource;

  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> queue; // descending order
  DynamicBitset checked;
  
  NaiveScheduler naiveScheduler;
};

#endif
