#ifndef panene_naive_data_source_h
#define panene_naive_data_source_h

#include <string>
#include <fstream>
#include "data_source.h"

namespace panene
{

class NaiveDataSource : public DataSource
{

public:
  NaiveDataSource() {
  }

  ~NaiveDataSource() {
    if(opened) delete[] data;
  }

  void open(const std::string& path, size_t n_, size_t d_) {
    n = n_;
    d = d_;
    data = new ElementType[n * d];    
    opened = true;
    
    std::ifstream ifs(path);

    for(size_t i = 0; i < n * d; ++i) {
      ifs >> data[i]; 
    }
  }

  inline ElementType* operator[](const IDType &id) const {
    return reinterpret_cast<ElementType *>(data + id * d);
  }

  ElementType* get(const IDType &id) const {
    return reinterpret_cast<ElementType *>(data + id * d);
  }

  DistanceType distL2Squared(const IDType &id1, const IDType &id2) const {
    DistanceType sum = 0;
    ElementType* ele1 = (*this)[id1];
    ElementType* ele2 = (*this)[id2];

    for(size_t i = 0; i < d; ++i)
      sum += (ele1[i] - ele2[i]) * (ele1[i] - ele2[i]);
    
    return sum;
  }

  DistanceType distL2Squared(const IDType &id1, const ElementType *p2) const {
    DistanceType sum = 0;
    ElementType* ele1 = (*this)[id1];

    for(size_t i = 0; i < d; ++i)
      sum += (ele1[i] - p2[i]) * (ele1[i] - p2[i]);
    
    return sum;
  }

  DistanceType sum(const IDType &id) const {
    DistanceType sum = 0;
    ElementType* ele = (*this)[id];

    for(size_t i = 0; i <d; ++i)
      sum += ele[i];

    return sum;
  }
  
  size_t size() const {
    return n;
  }

  size_t loaded() const {
    return n;
  }

  size_t dim() const {
    return d;
  }

	size_t n;
  size_t d;
  bool opened = false;
protected:
	ElementType* data;
};

}
#endif
