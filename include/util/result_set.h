#ifndef result_set_h_
#define result_set_h_

#include <cstdio>
#include <limits>
#include <vector>

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
    neighbors.resize(size);
    worstDist = (std::numeric_limits<DistanceType>::max)();

    for(size_t i = 0; i < size; ++i) {
      neighbors[i].id = 0;
      neighbors[i].dist = worstDist;
    }
  }

  const Neighbor<IDType, DistanceType> operator[](unsigned int index) const {
    return neighbors[index];
  }

  bool full() const
  {
      return worstDist < (std::numeric_limits<DistanceType>::max)();
  }

  friend std::ostream& operator<<( std::ostream& os, const ResultSet<IDType, DistanceType> &obj ) {
    for(size_t i = 0; i < obj.size; ++i) {
      os << i << ":" << obj.neighbors[i] << " ";
    }
    return os;  
  }

  void operator<<( const Neighbor<IDType, DistanceType> &neighbor ) {
    if(neighbor.dist >= worstDist) return;

    int i;
    for(i = size - 1; i >= 0; --i) {
      if(neighbors[i] == neighbor) return;
      if(neighbors[i] < neighbor) break;
    }

    // insert neighbor to (i + 1)
    size_t pos = i + 1;
 
    // shift (i+1) ~ size - 2

    for(size_t i = size - 1; i > pos; --i) {
      neighbors[i] = neighbors[i - 1];
    }

    neighbors[pos] = neighbor;
    worstDist = neighbors[size - 1].dist;
  }
  
  size_t size;
  DistanceType worstDist;
  std::vector<Neighbor<IDType, DistanceType>> neighbors;
};

}

#endif
