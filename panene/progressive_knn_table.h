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

#define DEBUG 1

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
    queued = DynamicBitset(dataSource.rows);
  }

  /*
  maxOps indicates the maximum number of operations that the function 'update' can execute.
  There are three types of operations:
    1) addNewPointOp adds a new point P to both table and index. It requires:
      An insertion operation to the index (O(lg N))
      An insertion operation to the table (O(1))
      A knn search for P (O(klg(N)))
      Mark the neighbors of P as dirty and insert them to the queue
    
    2) updateIndexOp updates a progressive k-d tree index.
      Basically, it calls the update function of the k-d tree and the update function
      creates a new k-d tree incrementally behind the scene

    2) updateTableOp takes a dirty point from a queue and updates its neighbors. It requires:
      A knn search for P
      Mark the neighbors of P as dirty and insert them to the queue


  Note that the three operations have different costs. 
  */
  
  UpdateResult update(size_t maxOps) {
    // To avoid keeping a copy of the whole data, we need to use an abstract dataframe that grows in real time.
    // Since we do not have such a dataframe currently, we assume all data are loaded in a datasource.
    
    // calculate the number of steps allocated for each operation
    Schedule schedule = naiveScheduler.schedule(maxOps);

#if DEBUG
    std::cerr << "[PKNNTable] Updating with a schedule " << schedule << std::endl;
#endif

    size_t oldSize = indexer.getSize();

    // 1. add new points to both index and table (addNewPointOps)
    
#if DEBUG
    std::cerr << "[PKNNTable] Adding points to the index" << std::endl;
#endif

    size_t addNewPointResult = indexer.addPoints(schedule.addNewPointOps);

#if DEBUG
    std::cerr << "[PKNNTable] " << addNewPointResult << " points have been newly indexed. " << indexer.getSize() << " points exist in trees." << std::endl;
#endif

    size_t size = indexer.getSize();

    // checks if points are added (if not, it means all points in the data have already been inserted)
    //
    if(addNewPointResult > 0) {
      assert(size > k); // check if at least k points are in the index

      Matrix<ElementType> newPoints(dataSource[size - addNewPointResult], addNewPointResult, dataSource.cols);

      Matrix<IDType> indices(new IDType[newPoints.rows * k], newPoints.rows, k);
      Matrix<DistanceType> dists(new DistanceType[newPoints.rows * k], newPoints.rows, k);

#if DEBUG
      std::cerr << "[PKNNTable] Filling in thw rows of the new points in KNNTable" << std::endl;
#endif 

      indexer.knnSearch(newPoints, indices, dists, k, searchParams);

#if DEBUG
      std::cerr << "[PKNNTable] KNNSearch Done" << std::endl;
#endif 

      std::vector<Neighbor> nns;
      nns.resize(k);

      for(unsigned int i = 0; i < newPoints.rows; ++i) {
        IDType id = oldSize + i;
        queued.set(id);

        for(unsigned int j = 0; j < k; ++j) {
          nns[j] = Neighbor(indices[i][j], dists[i][j]);

          if(!queued.test(indices[i][j])) {
            queued.set(indices[i][j]);
            queue.push(nns[j]);

#if DEBUG
            std::cerr << "[PKNNTable] Adding a dirty neighbor point to the queue " << nns[j] << std::endl;
#endif
          }
        }
        
        neighbors.push_back(nns);
      }

      delete[] indices.ptr();
      delete[] dists.ptr();
    }


    // 2. update the index (updateIndexOps)
    
#if DEBUG
    std::cerr << "[PKNNTable] Updating the index" << std::endl;
#endif
    size_t updateIndexResult = indexer.update(schedule.updateIndexOps);
    

#if DEBUG
    std::cerr << "[PKNNTable] Current cache table: " << std::endl;

    for(unsigned int i = 0; i < neighbors.size(); ++i) {
      for(unsigned int j = 0; j < k; ++j) {
        std::cerr << neighbors[i][j] << ' ';
      }
      std::cerr << std::endl;
    }

    std::cerr << "[PKNNTable] Starting processing queue" << std::endl;
#endif

    // 3. process the queue (updateTableOps)

    int checkCount = 0;
    Matrix<IDType> newIndices(new IDType[k], 1, k);
    Matrix<DistanceType> newDists(new DistanceType[k], 1, k);
    
    while((schedule.updateTableOps < 0 || checkCount < schedule.updateTableOps) && !queue.empty()) {
      auto q = queue.top();

#if DEBUG
      std::cerr << "[PKNNTable] Got a dirty point from the queue: " << q << std::endl;
#endif

      queue.pop();

      queued.reset(q.id);
      checkCount++;

      // we need to update the NN of q.id
      
      // get the new NN of the dirty point

      Matrix<ElementType> qvec(dataSource[q.id], 1, d);
#if DEBUG
      std::cerr << "[PKNNTable] Starting KNN search for the dirty point" << std::endl;
#endif
      indexer.knnSearch(qvec, newIndices, newDists, k, searchParams);
#if DEBUG
      std::cerr << "[PKNNTable] KNN search done" << std::endl;
#endif

      // check if there is a difference between previous NN and newly computed NN.
      
      unsigned int i;
      for(i = 0; i < k; ++i) {
        if(neighbors[q.id][i].id != newIndices[0][i])
          break;
      } 

      if(i < k) { // if there is a difference
        // then, mark the nn of q.id as dirty

        for(i = 0; i < k; ++i) {
          Neighbor ne(newIndices[0][i], newDists[0][i]);
          neighbors[q.id][i] = ne;

          if(!queued.test(ne.id)) {
            queued.set(ne.id);
            queue.push(ne);
          }
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
    std::cerr << "[PKNNTable] " << checkCount << " points have been updated." << std::endl;
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
  DynamicBitset queued;
  
  NaiveScheduler naiveScheduler;
};

#endif
