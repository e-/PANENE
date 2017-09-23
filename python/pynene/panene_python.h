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

  PyDataSource()
    : _d(0), _object(Py_None) {
  }

  ~PyDataSource() {
    set_array(Py_None);
  }

  void set_array(PyObject * o) {
    if (o == _object) return;
    Py_INCREF(o);
    Py_DECREF(_object);
    _object = o;
    set_dim();
  }

  PyObject * get_array() const {
    return _object;
  }

  ElementType get(const IDType &id, const IDType &dim) const {
    PyObject *key1 = PyInt_FromLong(id);
    PyObject *key2 = PyInt_FromLong(dim);
    PyObject *tuple = PyTuple_Pack(2, key1, key2);
    PyObject *pf = PyObject_GetItem(_object, tuple);
    ElementType ret = (ElementType)PyFloat_AsDouble(pf);
    Py_DECREF(tuple);
    Py_DECREF(pf);
    return ret;
  }

  IDType findDimWithMaxSpan(const IDType &id1, const IDType &id2) {
    size_t dimension = 0;
    ElementType maxSpan = 0;
    size_t d = dim();

    for(size_t i = 0; i < d; ++i) {
      ElementType span = std::abs(this->get(id1, i) - this->get(id2, i));
      if(maxSpan < span) {
        maxSpan = span;
        dimension = i;
      }
    }

    return dimension;
  }

  void computeMeanAndVar(const IDType *ids, int count, std::vector<DistanceType> &mean, std::vector<DistanceType> &var) {
    size_t d = dim();
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
    size_t d = dim();

    for(size_t i = 0; i < d; ++i) {
      ElementType v1 = this->get(id1, i), v2 = this->get(id2, i);
      sum += (v1 - v2) * (v1 - v2);
    }
    
    return sum;
  }

  size_t size() const {
    if (_object==Py_None) return 0;
    return PyObject_Length(_object);
  }

  size_t loaded() const {
    return size();
  }

  size_t dim() const {
    return _d;
  }

 protected:  
  int _d;
  PyObject * _object;

  void set_dim() {
    if (_object==Py_None) {
      _d = 0;
    }
    else {
      PyObject * shape = PyObject_CallMethod(_object, "shape", NULL);
      PyObject * dim = PyTuple_GetItem(shape, 1);
      if (!PyLong_Check(dim))
        _d = 0;
      else {
        _d = (int)PyLong_AsLong(dim);
      }
      Py_DECREF(shape);
    }
  }

};

typedef Neighbor<size_t, float> PyNeighbor;

typedef ResultSet<size_t, float> PyResultSet;

typedef ProgressiveKDTreeIndex<L2<float>, PyDataSource> PyIndexL2;
