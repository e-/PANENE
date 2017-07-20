#ifndef panene_progressive_knn_table_h
#define panene_progressive_knn_table_h

#include <vector>
#include <iostream>
#include <queue>
#include <cassert>

#include <progressive_kd_tree_index.h>
#include <scheduler/scheduler.h>
#include <functional>

//#define DEBUG 1

namespace panene {

template <typename Indexer, typename DataSource>
class ProgressiveKNNTable {
  typedef typename DataSource::ElementType ElementType;
  typedef typename DataSource::DistanceType DistanceType;
  typedef typename DataSource::IDType IDType;
  
  typedef Neighbor<IDType, DistanceType> NeighborType;

public:
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

  void setDataSource(DataSource *dataSource_) {
    dataSource = dataSource_;
    indexer.setDataSource(dataSource);
    queued = DynamicBitset(dataSource -> size());
  }

  void setScheduler(Scheduler *scheduler) {
    this -> scheduler = scheduler;
  }

  size_t getSize() {
    return indexer.getSize();
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
    Schedule schedule = scheduler->schedule(maxOps);

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

      std::vector<IDType> newPoints(addNewPointResult);
      
      for(size_t i = 0; i < addNewPointResult; ++i) {
        newPoints[i] = oldSize + i;
      } 

      std::vector<ResultSet<IDType, DistanceType>> results(addNewPointResult);
      for(size_t i = 0; i < addNewPointResult; ++i)
        results[i] = ResultSet<IDType, DistanceType>(k);

#if DEBUG
      std::cerr << "[PKNNTable] Filling in thw rows of the new points in KNNTable" << std::endl;
#endif 

      indexer.knnSearch(newPoints, results, k, searchParams);

#if DEBUG
      std::cerr << "[PKNNTable] KNNSearch Done" << std::endl;
#endif 

      for(size_t i = 0; i < addNewPointResult; ++i) {
        IDType id = oldSize + i;
        queued.set(id);

        for(IDType j = 0; j < k; ++j) {
          if(!queued.test(results[i][j].id)) {
            queued.set(results[i][j].id);
            queue.push(results[i][j]);

#if DEBUG
            std::cerr << "[PKNNTable] Adding a dirty neighbor point to the queue " << nns[j] << std::endl;
#endif
          }
        }
        
        neighbors.push_back(results[i]);
      }
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
      ResultSet<IDType, DistanceType> result(k);
#if DEBUG
      std::cerr << "[PKNNTable] Starting KNN search for the dirty point" << std::endl;
#endif
      indexer.knnSearch(q.id, result, k, searchParams);
#if DEBUG
      std::cerr << "[PKNNTable] KNN search done" << std::endl;
#endif

      // check if there is a difference between previous NN and newly computed NN.      
      size_t i;
      for(i = 0; i < k; ++i) {
        if(neighbors[q.id][i] != result[i])
          break;
      } 

      if(i < k) { // if there is a difference
        // then, mark the nn of q.id as dirty

        neighbors[q.id] = result;

        for(i = 0; i < k; ++i) {
          if(!queued.test(result[i].id)) {
            queued.set(result[i].id);
            queue.push(result[i]);
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

#if DEBUG
    std::cerr << "[PKNNTable] " << checkCount << " points have been updated." << std::endl;
#endif
    
    size_t updateTableResult = checkCount;

    return UpdateResult(schedule, addNewPointResult, updateIndexResult, updateTableResult);
  };

  ResultSet<IDType, DistanceType> &getNeighbors(const IDType id) {
    return neighbors[id];
  }

  Indexer indexer;

private:
  unsigned int d;
  unsigned int k;
  SearchParams searchParams;
  std::vector<ResultSet<IDType, DistanceType>> neighbors;
  DataSource *dataSource;
  Scheduler *scheduler;

  std::priority_queue<NeighborType, std::vector<NeighborType>, std::greater<NeighborType>> queue; // descending order
  DynamicBitset queued;
};

}
#endif
