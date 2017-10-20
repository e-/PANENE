#ifndef panene_data_source_h
#define panene_data_source_h

#define USE_DATA_SOURCE_SYMBOLS public: \
  typedef T IDType;\
  typedef D Distance;\
  typedef typename D::ElementType ElementType;\
  typedef typename D::ResultType DistanceType;\

namespace panene
{

template<typename T, class D>
class DataSource
{

public:
  typedef T IDType;
  typedef D Distance;
  typedef typename D::ElementType ElementType;
  typedef typename D::ResultType DistanceType;

  virtual inline ElementType get(const IDType &id, const IDType &dim) const = 0;
  virtual void get(const IDType &id, std::vector<ElementType> &result) const = 0;

  virtual IDType findDimWithMaxSpan(const IDType &id1, const IDType &id2) = 0;
  virtual void computeMeanAndVar(const IDType *ids, int count, std::vector<DistanceType> &mean, std::vector<DistanceType> &var) = 0;

  virtual DistanceType getSquaredDistance(const IDType &id1, const IDType &id2) const = 0;

  virtual size_t size() const = 0;
  virtual size_t loaded() const = 0;
  virtual size_t dim() const = 0;
};
}

#endif
