#ifndef result_set_h_
#define result_set_h_

#include <cstdio>
#include <limits>
#include <vector>
#include <iostream>

namespace panene {

template <typename IDType, typename DistanceType>
struct Neighbor {
  IDType id;    
  DistanceType dist;

  Neighbor() = default;
  Neighbor(IDType id_, DistanceType dist_) : id(id_), dist(dist_) {}

  friend std::ostream& operator<<( std::ostream& os, const Neighbor<IDType, DistanceType>& obj ) {
    os << "(" << obj.id << ", " << obj.dist << ")";
    return os;  
  }

  bool operator< (const Neighbor &n) const {
    return this->dist < n.dist;
  }

  bool operator> (const Neighbor &n) const {
    return this->dist > n.dist;
  }

  bool operator== (const Neighbor &n) const {
    return this->id == n.id;
  }
  
  bool operator!= (const Neighbor &n) const {
    return !(this->id == n.id);
  }
};

template <typename IDType, typename DistanceType>
struct ResultSet {
  ResultSet() = default;
  ResultSet(size_t size_) : size(size_) {
    nn.resize(size);
    worstDist = (std::numeric_limits<DistanceType>::max)();

    for(size_t i = 0; i < size; ++i) {
      nn[i].id = -1;
      nn[i].dist = worstDist;
    }
  }

  const Neighbor<IDType, DistanceType> operator[](IDType id) const {
    return nn[id];
  }

  const std::vector<IDType> neighbors(IDType id) {
    std::vector<IDType> res(size);
    
    for(size_t i = 0; i < size; ++i)
      res[i] = nn[id].id;
    
    return res;
  }

  const std::vector<DistanceType> distances(IDType id) {
    std::vector<DistanceType> res(size);
    
    for(size_t i = 0; i < size; ++i)
      res[i] = nn[id].dist;
    
    return res;
  }

  bool full() const
  {
      return worstDist < (std::numeric_limits<DistanceType>::max)();
  }

  friend std::ostream& operator<<( std::ostream& os, const ResultSet<IDType, DistanceType> &obj ) {
    for(size_t i = 0; i < obj.size; ++i) {
      os << i << ":" << obj.nn[i] << " ";
    }
    return os;  
  }

  void operator<<( const Neighbor<IDType, DistanceType> &neighbor ) {
    if(neighbor.dist >= worstDist) return;

    int i;
    for(i = size - 1; i >= 0; --i) {
      if(nn[i] == neighbor) return;
      if(nn[i] < neighbor) break;
    }

    // insert neighbor to (i + 1)
    size_t pos = i + 1;
 
    // shift (i+1) ~ size - 2

    for(size_t i = size - 1; i > pos; --i) {
      nn[i] = nn[i - 1];
    }

    nn[pos] = neighbor;
    worstDist = nn[size - 1].dist;
  }
  
  size_t size;
  DistanceType worstDist;
  std::vector<Neighbor<IDType, DistanceType>> nn;
};

}

#endif
