#ifndef panene_progressive_knn_table_h
#define panene_progressive_knn_table_h

#include <vector>
#include <iostream>
#include <queue>

#ifdef BENCHMARK
#include <tests/metadata.h>
#endif

#include <schedule.h>
#include <progressive_kd_tree_index.h>
#include <functional>

//#define DEBUG 1

namespace panene {

struct WeightSet {
  float addPointsWeight;
  float updateIndexWeight;
  float updateTableWeight;

  WeightSet(float addPointsWeight_, float updateIndexWeight_) : addPointsWeight(addPointsWeight_), updateIndexWeight(updateIndexWeight_) {
    this->updateTableWeight = 1 - addPointsWeight - updateIndexWeight;
  }
};

struct UpdateResult {
  Schedule schedule;
  size_t addPointsResult;
  size_t updateIndexResult;
  size_t updateTableResult;
  size_t numPointsInserted;

  double addPointsElapsed;
  double updateIndexElapsed;
  double updateTableElapsed;

  UpdateResult(const Schedule &schedule_, size_t addPointsResult_, size_t updateIndexResult_, size_t updateTableResult_, size_t numPointsInserted_,
               double addPointsElapsed_, double updateIndexElapsed_, double updateTableElapsed_) :
    schedule(schedule_), addPointsResult(addPointsResult_), updateIndexResult(updateIndexResult_), updateTableResult(updateTableResult_),
    numPointsInserted(numPointsInserted_), addPointsElapsed(addPointsElapsed_), updateIndexElapsed(updateIndexElapsed_), updateTableElapsed(updateTableElapsed_)
  {
  }

  friend std::ostream& operator<<(std::ostream& os, const UpdateResult& obj) {
    os << "UpdateResult(addNewPointOps: " << obj.addPointsResult << " / " << obj.schedule.addNewPointOps << ", "
      << "updateIndexOps: " << obj.updateIndexResult << " / " << obj.schedule.updateIndexOps << ", "
      << "updateTableOps: " << obj.updateTableResult << " / " << obj.schedule.updateTableOps << ")";
    return os;
  }
};

template <typename Indexer, typename DataSource>
class ProgressiveKNNTable {
  typedef typename DataSource::ElementType ElementType;
  typedef typename DataSource::DistanceType DistanceType;
  typedef typename DataSource::IDType IDType;
  
  typedef Neighbor<IDType, DistanceType> NeighborType;

public:  

  ProgressiveKNNTable(size_t k_, size_t d_, IndexParams indexParams_, SearchParams searchParams_, WeightSet weights_) : 
    d(d_), k(k_), indexer(Indexer(indexParams_)), weights(weights_), searchParams(searchParams_){

    numPointsInserted = 0;
  }

  void setDataSource(DataSource *dataSource_) {
    dataSource = dataSource_;
    indexer.setDataSource(dataSource);
    queued = DynamicBitset(dataSource -> size());
  }

  /*void setScheduler(Scheduler *scheduler) {
    this -> scheduler = scheduler;
  }*/

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
        
    // 1. add new points to both index and table (addPointsOps)

    size_t addPointsOps = 0, updateIndexOps = 0;
    double addPointsElapsed = 0;
    
#ifdef BENCHMARK
    Timer timer;
    timer.begin();
#endif
   
    if (indexer.updateStatus == NoUpdate) {
      addPointsOps = maxOps * (weights.addPointsWeight + weights.updateIndexWeight);
    }
    else {
      addPointsOps = maxOps * weights.addPointsWeight;
      updateIndexOps = maxOps * weights.updateIndexWeight;
    }    

#if DEBUG
    std::cerr << "[PKNNTable] Adding points to the index" << std::endl;
#endif

    size_t addPointsResult = 0;
    
    if (addPointsOps > 0) {
      addPointsResult = indexer.addPoints(addPointsOps);
      numPointsInserted += addPointsResult;
    }


    if (addPointsResult == 0) { // all points are inserted to the index
      weights.updateIndexWeight += weights.addPointsWeight / 2;
      weights.addPointsWeight = 0;
      weights.updateTableWeight = 1 - weights.updateTableWeight;
    }

#if DEBUG
    std::cerr << "[PKNNTable] " << addNewPointResult << " points have been newly indexed. " << indexer.getSize() << " points exist in trees." << std::endl;
#endif

    size_t size = indexer.getSize();
    size_t oldSize = size - addPointsResult;

    // checks if points are added (if not, it means all points in the data have already been inserted)
    
    if(addPointsResult > 0) {  

      std::vector<IDType> newPoints(addPointsResult);
      
      for(size_t i = 0; i < addPointsResult; ++i) {
        newPoints[i] = oldSize + i;
      } 

      std::vector<ResultSet<IDType, DistanceType>> results(addPointsResult);
      for(size_t i = 0; i < addPointsResult; ++i)
        results[i] = ResultSet<IDType, DistanceType>(k);

#if DEBUG
      std::cerr << "[PKNNTable] Filling in thw rows of the new points in KNNTable" << std::endl;
#endif 

      indexer.knnSearch(newPoints, results, k, searchParams);

#if DEBUG
      std::cerr << "[PKNNTable] KNNSearch Done" << std::endl;
#endif 

      for(size_t i = 0; i < addPointsResult; ++i) {
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
#ifdef BENCHMARK
    addPointsElapsed = timer.end();
#endif

    // 2. update the index (updateIndexOps)
    
#if DEBUG
    std::cerr << "[PKNNTable] Updating the index" << std::endl;
#endif
    
    size_t updateIndexResult = 0;
    double updateIndexElapsed = 0;

#ifdef BENCHMARK
    timer.begin();
#endif

    if (updateIndexOps > 0) {
      updateIndexResult = indexer.update(updateIndexOps);
    }

    
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

#ifdef BENCHMARK
    updateIndexElapsed = timer.end();
#endif

    // 3. process the queue (updateTableOps)

    size_t updateTableOps = maxOps - addPointsOps - updateIndexOps;
    double updateTableElapsed = 0;

#ifdef BENCHMARK
    timer.begin();
#endif 

    int checkCount = 0;    
    while(checkCount < updateTableOps && !queue.empty()) {
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
#ifdef BENCHMARK
    updateTableElapsed = timer.end();
#endif

#ifdef BENCHMARK
    return UpdateResult(Schedule(addPointsOps, updateIndexOps, updateTableOps), addPointsResult, updateIndexResult, updateTableResult, numPointsInserted,
      addPointsElapsed, updateIndexElapsed, updateTableElapsed);
#else
    return UpdateResult(Schedule(addPointsOps, updateIndexOps, updateTableOps), addPointsResult, updateIndexResult, updateTableResult, numPointsInserted, 0, 0, 0);
#endif
  };

  ResultSet<IDType, DistanceType> &getNeighbors(const IDType id) {
    return neighbors[id];
  }

  Indexer indexer;

private:
  size_t d;
  size_t k;
  SearchParams searchParams;
  std::vector<ResultSet<IDType, DistanceType>> neighbors;
  DataSource *dataSource;
  WeightSet weights;
  size_t numPointsInserted;

  std::priority_queue<NeighborType, std::vector<NeighborType>, std::greater<NeighborType>> queue; // descending order
  DynamicBitset queued;
};

}
#endif
