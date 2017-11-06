#ifndef panene_vector_data_sink_h
#define panene_vector_data_sink_h

#include <vector>
#include <cassert>

namespace panene {

  template <typename IDType, typename DistanceType>
class VectorDataSink {
public:
  VectorDataSink(size_t size_, size_t k_): size(size_), k(k_) {
    neighbors.resize(size_);
    distances.resize(size_);

    for(size_t i = 0; i < size_; ++i) { 
      neighbors[i].reserve(k);
      distances[i].reserve(k);
    }
  }

  const IDType * getNeighbors(IDType id) const {
    return &neighbors[id][0];
  }

  const DistanceType * getDistances(IDType id) const {
    return &distances[id][0];
  }

  void setNeighbors(IDType id, const IDType * neighbors_, const DistanceType * distances_) {
    // we "copy" the neighbors and distances 
    for(size_t i = 0; i < k; ++i) {
      neighbors[id][i] = neighbors_[i];
      distances[id][i] = distances_[i];
    }
  }

private:
  size_t size;
  size_t k;

  std::vector<std::vector<IDType>> neighbors;
  std::vector<std::vector<DistanceType>> distances;
};

};

#endif
