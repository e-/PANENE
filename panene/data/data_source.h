#ifndef panene_data_source_h
#define panene_data_source_h

#include <string>
#include <vector>

namespace panene
{

class DataSource
{
public:
	typedef size_t IDType;
  typedef float ElementType;
  typedef float DistanceType;

  // opens a dataset of n points of d dimensions
  virtual void open(const std::string& path, size_t n, size_t d) = 0;

  // returns a point with the given id
  inline virtual ElementType* operator[](const IDType &id) const = 0;

  // same with operator[]
  virtual ElementType* get(const IDType &id) const = 0;

  // returns the L1 distance between two points
  virtual std::vector<ElementType> subtract(const IDType &id1, const IDType &id2) const = 0;

  // returns the L1 distance between two points
  virtual std::vector<ElementType> subtract(const IDType &id1, const ElementType *p2) const = 0;
  
  // returns the squared distance between two points
  virtual DistanceType distL2Squared(const IDType &id1, const IDType &id2) const = 0;

  // returns the squared distance between two points
  virtual DistanceType distL2Squared(const IDType &id1, const ElementType *p2) const = 0;

  // returns the number of points in the dataset
  virtual size_t size() const = 0;

  // returns the number of points that are loaded and thus can be accessed
  virtual size_t loaded() const = 0;

  // returns the dimensionality of the dataset
  virtual size_t dim() const = 0;
};

}

#endif
