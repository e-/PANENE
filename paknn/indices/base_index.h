#ifndef base_index_h
#define base_index_h

#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdio>
#include <iostream>

#include "../util/matrix.h"
#include "../util/allocator.h"
#include "../util/result_set.h"
#include "../util/heap.h"
#include "../util/dynamic_bitset.h"
#include "../util/random.h"
#include "dist.h"

namespace paknn
{

struct IndexParams {
  int trees;

  IndexParams(int trees_) : trees(trees_) {}
};

struct SearchParams {
  int checks; // 32
  float eps; // 0
  int sorted;
  int cores;
  
  SearchParams() = default;
  SearchParams(int checks_, float eps_ = 0, int sorted_ = 0, int cores_ = 0) : checks(checks_), eps(eps_), sorted(sorted_), cores(cores_) {}
};

template <typename Distance>
class BaseIndex
{
public:
  typedef float ElementType;
  typedef float DistanceType;

  virtual void addPoints(size_t end) = 0;
  virtual size_t getSize() = 0;

  virtual int usedMemory() const = 0;

  virtual int knnSearch(const Matrix<ElementType>& queries,
			Matrix<size_t>& indices,
			Matrix<DistanceType>& dists,
			size_t knn,
			const SearchParams& params) const = 0;

  virtual void freeIndex() = 0;
};
}

#endif
