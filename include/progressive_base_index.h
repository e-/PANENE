#ifndef panene_progressive_base_index_h
#define panene_progressive_base_index_h

#include <vector>
#include <algorithm>
#include <random>

#include <util/allocator.h>
#include <util/result_set.h>
#include <util/heap.h>
#include <util/dynamic_bitset.h>
#include <util/random.h>
#include <util/result_set.h>
#include <dist.h>

namespace panene
{

struct IndexParams {
  int trees;

  IndexParams(int trees_ = 4) : trees(trees_) {}
};

struct SearchParams {
  int checks; // 32
  float eps; // 0
  int sorted;
  int cores;
  
  SearchParams(int checks_ = 32, float eps_ = 0, int sorted_ = 0, int cores_ = 0) : checks(checks_), eps(eps_), sorted(sorted_), cores(cores_) {}
};

template <typename Distance, typename DataSource>
class ProgressiveBaseIndex
{
public:
  typedef typename DataSource::ElementType ElementType;
  typedef typename DataSource::DistanceType DistanceType;
  typedef typename DataSource::IDType IDType;
  
  virtual size_t addPoints(size_t end) = 0;
  virtual size_t update(int ops) = 0;

  virtual size_t getSize() = 0;

  virtual int usedMemory() const = 0;

  virtual void knnSearch(
      const IDType &qid,
      ResultSet<IDType, DistanceType> &resultSet,
      size_t knn,
      const SearchParams& params) const = 0;

  virtual void knnSearch(
      const std::vector<IDType> qids,
      std::vector<ResultSet<IDType, DistanceType>> &resultSet,
      size_t knn,
      const SearchParams& params) const = 0;

  virtual void freeIndex() = 0;
};

}

#endif
