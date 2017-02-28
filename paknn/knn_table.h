#ifndef knn_table_h
#define knn_table_h

#include <flann/flann.hpp>
#include <vector>
#include <iostream>

template <class Indexer>
class KNNTable {
  typedef typename Indexer::ElementType ElementType;
  typedef typename Indexer::DistanceType DistanceType;
  typedef long unsigned int IDType;
  
public:
  struct Neighbor {
    ElementType index;
    DistanceType distance;

    Neighbor(ElementType index_, DistanceType distance_) : index(index_), distance(distance_) {}
    friend std::ostream& operator<<( std::ostream& os, const Neighbor& obj ) {
      os << "(" << obj.index << ", " << obj.distance << ")";
      return os;  
    }
  };

  KNNTable(unsigned int k_, unsigned int d_, flann::IndexParams indexParams_, flann::SearchParams searchParams_) : d(d_), k(k_), indexer(flann::KDTreeIndexParams(1)) {
    flann::Matrix<DistanceType> initData(nullptr, 0, d_);
    indexer = Indexer(initData, indexParams_);
    searchParams = searchParams_;
  }
  
  void addPoints(flann::Matrix<DistanceType> points) { 
    // 1. index new points
    
    indexer.addPoints(points);
    std::cout << indexer.veclen() << ' ' << indexer.size() << std::endl;

    assert(indexer.size() > k);

    // 2. compute new knn
  
    flann::Matrix<IDType> indices(new IDType[points.rows * k], points.rows, k);
    flann::Matrix<DistanceType> dists(new DistanceType[points.rows * k], points.rows, k);
                 
    indexer.knnSearch(points, indices, dists, k, searchParams);

    for(unsigned int i = 0; i < points.rows; ++i) {
      std::vector<Neighbor> nns;
      for(unsigned int j = 0; j < k; ++j) {
        nns.push_back(Neighbor(indices[i][j], dists[i][j]));
      }
      
      neighbors.push_back(nns);
    }

    delete[] indices.ptr();
    delete[] dists.ptr();

    for(unsigned int i = 0; i < neighbors.size(); ++i) {
      for(unsigned int j = 0; j < k; ++j) {
        std::cout << neighbors[i][j] << ' ';
      }
      std::cout << std::endl;
    }
    // 3. update old knn

  };

private:
  flann::Matrix<DistanceType> data;
  unsigned int d;
  unsigned int k;
  Indexer indexer;
  flann::SearchParams searchParams;
  std::vector<std::vector<Neighbor>> neighbors;
};

#endif
