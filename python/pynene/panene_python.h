#include <Python.h>

#include <progressive_knn_table.h>
#include <naive_data_source.h>

using namespace panene;

class PyDataSource
{
 public:

  typedef size_t IDType;
  typedef float ElementType;
  typedef float DistanceType;

  PyDataSource() {
    object = Py_None;
  }

  ~PyDataSource() {
    set_array(Py_None);
  }

  void set_array(PyObject * o) {
    if (o == object) return;
    Py_XINCREF(o);
    Py_XDECREF(object);
    object = o;
  }

  PyObject * get_array() const {
    return object;
  }

  ElementType get(const IDType &id, const IDType &dim) const {
    PyObject *key1 = PyInt_FromLong(id);
    PyObject *key2 = PyInt_FromLong(dim);
    PyObject *tuple = PyTuple_Pack(2, key1, key2);
    PyObject *pf = PyObject_GetItem(object, tuple);
    ElementType ret = (ElementType)PyFloat_AsDouble(pf);
    Py_DECREF(tuple);
    Py_DECREF(pf);
    return ret;
  }

  IDType findDimWithMaxSpan(const IDType &id1, const IDType &id2) {
    size_t dim = 0;
    ElementType maxSpan = 0;

    for(size_t i = 0; i < d; ++i) {
      ElementType span = std::abs(this->get(id1, i) - this->get(id2, i));
      if(maxSpan < span) {
        maxSpan = span;
        dim = i;
      }
    }

    return dim;
  }

  void computeMeanAndVar(const IDType *ids, int count, std::vector<DistanceType> &mean, std::vector<DistanceType> &var) {
    mean.resize(d);
    var.resize(d);

    for (size_t i = 0; i < d; ++i) 
      mean[i] = var[i] = 0;

    for (int j = 0; j < count; ++j) {
      for (size_t i = 0; i < d; ++i) {
        mean[i] += this->get(ids[j], i);
      }
    }

    DistanceType divFactor = DistanceType(1)/count;

    for (size_t i = 0 ; i < d; ++i) {
      mean[i] *= divFactor;
    }

    /* Compute variances */
    for (int j = 0; j < count; ++j) {
      for (size_t i = 0; i < d; ++i) {
        DistanceType dist = this->get(ids[j], i) - mean[i];
        var[i] += dist * dist;
      }
    }

    for(size_t i = 0; i < d; ++i) {
      var[i] *= divFactor;
    }
  }  

  DistanceType distL2Squared(const IDType &id1, const IDType &id2) const {
    DistanceType sum = 0;

    for(size_t i = 0; i < d; ++i) {
      ElementType v1 = this->get(id1, i), v2 = this->get(id2, i);
      sum += (v1 - v2) * (v1 - v2);
    }
    
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
  PyObject * object;
};

typedef Neighbor<size_t, float> PyNeighbor;

typedef ResultSet<size_t, float> PyResultSet;

typedef ProgressiveKDTreeIndex<L2<float>, PyDataSource> PyIndexL2;
