#ifndef knn_table_h
#define knn_table_h

#include <flann/flann.hpp>
#include <vector>
#include <iostream>
#include <queue>

#define MAX_CHECK 0
#define DEBUG 0

template <class Indexer>
class KNNTable {
  typedef typename Indexer::ElementType ElementType;
  typedef typename Indexer::DistanceType DistanceType;
  typedef long unsigned int IDType;
  
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

  KNNTable(unsigned int k_, unsigned int d_, flann::IndexParams indexParams_, flann::SearchParams searchParams_) : d(d_), k(k_), indexer(flann::KDTreeIndexParams(1)) {
    flann::Matrix<DistanceType> initData(nullptr, 0, d_);
    indexer = Indexer(initData, indexParams_);
    searchParams = searchParams_;
  }

  void extendDataset(const flann::Matrix<DistanceType>& new_points) { 
    size_t old_size = points.size();
		size_t new_size = points.size() + new_points.rows;

    points.resize(new_size);

    for (size_t i = old_size; i < new_size; ++i) {
      points[i] = new_points[i - old_size];
    }
  }

  void addPoints(const flann::Matrix<DistanceType>& new_points) { 
    // 0. add new points to data

    extendDataset(new_points);

    // 1. index new points
    
    size_t old_size = indexer.size();

    indexer.addPoints(new_points);
#if DEBUG
    std::cout << new_points.rows << " new points have been indexed. " << indexer.size() << " points exist in trees." << std::endl;
#endif

    assert(indexer.size() > k); // check if at least k points are in the index

    size_t size = indexer.size();

    // 2. compute new knn
  
    flann::Matrix<IDType> indices(new IDType[new_points.rows * k], new_points.rows, k);
    flann::Matrix<DistanceType> dists(new DistanceType[new_points.rows * k], new_points.rows, k);
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> queue; // descending order
    flann::DynamicBitset checked(size);

    indexer.knnSearch(new_points, indices, dists, k, searchParams);

    std::vector<Neighbor> nns;
    nns.resize(k);

    for(unsigned int i = 0; i < new_points.rows; ++i) {
      IDType id = old_size + i;
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
#endif

    // 3. update old knn

    unsigned int checkCount = 0;
    flann::Matrix<IDType> new_indices(new IDType[k], 1, k);
    flann::Matrix<DistanceType> new_dists(new DistanceType[k], 1, k);

    while(!queue.empty()) {
      auto q = queue.top();
      queue.pop();

      if(checked.test(q.id) || (MAX_CHECK > 0 && checkCount > MAX_CHECK)) continue;
      checked.set(q.id);
      checkCount++;

      // we need to update the NN of q.id
      
      // get new NN
      
      flann::Matrix<ElementType> qvec(points[q.id], 1, d);
      indexer.knnSearch(qvec, new_indices, new_dists, k, searchParams);
      
      // check if there is a difference between previous NN and newly computed NN.
      
      unsigned int i;
      for(i = 0; i < k; ++i) {
        if(neighbors[q.id][i].id != new_indices[0][i])
          break;
      } 

      if(i < k) { // if there is a difference
        // update q.id
        for(i = 0; i < k; ++i) {
          Neighbor ne(new_indices[0][i], new_dists[0][i]);
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
        std::cout << '(' << new_indices[0][i] << ", " << new_dists[0][i] << ") ";
      }
      std::cout << std::endl;
#endif
    }

    delete[] new_indices.ptr();
    delete[] new_dists.ptr();

    std::cout << checkCount << '\t';

#if DEBUG
    std::cout << checkCount << " points have been updated." << std::endl;
#endif
  };

  std::vector<Neighbor>& getNeighbors(const IDType id) {
    return neighbors[id];
  }

private:
  std::vector<ElementType*> points;
  unsigned int d;
  unsigned int k;
  Indexer indexer;
  flann::SearchParams searchParams;
  std::vector<std::vector<Neighbor>> neighbors;
};

#endif
