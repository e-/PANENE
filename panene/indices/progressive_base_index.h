#ifndef panene_progressive_base_index_h
#define panene_progressive_base_index_h

#include "base_index.h"

namespace panene
{

template <typename Distance>
class ProgressiveBaseIndex
{
public:
  typedef float ElementType;
  typedef float DistanceType;

  virtual size_t addPoints(size_t end) = 0;
  virtual size_t update(int ops) = 0;

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
