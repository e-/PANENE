#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include <progressive_knn_table.h>

#ifndef NDEBUG
#include <iostream>
#define DBG(x) x
#else
#define DBG(x)
#endif


using namespace panene;


class PyDataSource
{
 public:

  typedef size_t IDType;
  typedef float ElementType;
  typedef float DistanceType;
  typedef L2<float> Distance;

  PyDataSource(PyObject * o)
    : _d(0), _object(Py_None), _array(nullptr) {
    Py_INCREF(_object);
    import_array_wrap();
    set_array(o);
  }


#if PY_VERSION_HEX >= 0x03000000
  void* import_array_wrap()
#else
  void  import_array_wrap()
#endif
  {
    import_array(); // required to avoid core dumps from numpy
    // I wrote a wrapper for the macro import_array() since it 'returns' when there's an error but the compilers do not allow the use of a 'return' keyword in a class constructor. 
  }

  ~PyDataSource() {
    DBG(std::cerr << "calling destructor" << std::endl);
    if (_object != nullptr) {
      DBG(std::cerr << "~ _object refcount: " << _object->ob_refcnt << std::endl);
      Py_DECREF(_object);
    }
    _object = nullptr;
    _array = nullptr;
  }

  void set_array(PyObject * o) {
    DBG(std::cerr << "set_array(" << o << ")" << std::endl;)
    if (o == _object) return;
    Py_INCREF(o);
    Py_DECREF(_object);
    _object = o;
    _array = nullptr;
    DBG(std::cerr << "set_array _object refcount: " << _object->ob_refcnt << std::endl);
    if (_object != Py_None) {
      if(PyArray_Check(_object)) {
        DBG(std::cerr << "Object is an array..." << std::endl;)
        PyArrayObject * array = (PyArrayObject*)_object;
        if (PyArray_IS_C_CONTIGUOUS(array)
            && (PyArray_TYPE(array) == NPY_FLOAT)) {
          DBG(std::cerr << "acceptable for fast get" << std::endl;)
          _array = array;
        }
        else {
          DBG(std::cerr << "not acceptable for fast get" << std::endl;)
        }
      } 
      else {
        DBG(std::cerr << "Object is not an array...";)
      }
    }
    else {
      DBG(std::cerr << "Object is not an array...";)
    }

    set_dim();
  }

  PyObject * get_array() const {
    Py_INCREF(_object); //TODO check that this is the right way to do it
    return _object;
  }

  ElementType get(const IDType &id, const IDType &dim) const {
    if (_array != nullptr) {
      //DBG(std::cerr << "get from array" << std::endl);
      return *(float *) PyArray_GETPTR2(_array, id, dim);
    }
    DBG(std::cerr << "Starting get(" << id << "," << dim << ")" << std::endl;)
    PyObject *key1 = PyInt_FromLong(id);
    //DBG(std::cerr << "Created key1" << std::endl;)
    PyObject *key2 = PyInt_FromLong(dim);
    //DBG(std::cerr << "Created key2" << std::endl;)
    PyObject *tuple = PyTuple_Pack(2, key1, key2);
    PyObject *pf = PyObject_GetItem(_object, tuple);
    ElementType ret = 0;
    DBG(std::cerr << "Got item " << pf;)
    if (pf != nullptr) {
      DBG(std::cerr << " type: " << pf->ob_type->tp_name << std::endl;)
      ret = (ElementType)PyFloat_AsDouble(pf);
      Py_DECREF(pf);
    }
    else {
      DBG(std::cerr << " not a number " << std::endl);
    }
    DBG(std::cerr << " value is: " << ret << std::endl);
    Py_DECREF(tuple);
    return ret;
  }

  IDType findDimWithMaxSpan(const IDType &id1, const IDType &id2) {
    size_t dimension = 0;
    ElementType maxSpan = 0;
    size_t d = dim();

    for(size_t i = 0; i < d; ++i) {
      ElementType span = std::abs(get(id1, i) - get(id2, i));
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
        mean[i] += get(ids[j], i);
      }
    }

    DistanceType divFactor = DistanceType(1)/count;

    for (size_t i = 0 ; i < d; ++i) {
      mean[i] *= divFactor;
    }

    /* Compute variances */
    for (int j = 0; j < count; ++j) {
      for (size_t i = 0; i < d; ++i) {
        DistanceType dist = get(ids[j], i) - mean[i];
        var[i] += dist * dist;
      }
    }

    for(size_t i = 0; i < d; ++i) {
      var[i] *= divFactor;
    }
  }  

  DistanceType getSquaredDistance(const IDType &id1, const IDType &id2) const {
    DistanceType sum = 0;
    size_t d = dim();

    for(size_t i = 0; i < d; ++i) {
      ElementType v1 = get(id1, i), v2 = get(id2, i);
      sum += (v1 - v2) * (v1 - v2);
    }
    
    return sum;
  }

  DistanceType getSquaredDistance(const IDType &id1, const std::vector<ElementType> &vec2) const {
    DistanceType sum = 0;
    size_t d = dim();

    for(size_t i = 0; i < d; ++i) {
      ElementType v1 = get(id1, i), v2 = vec2[i];
      sum += (v1 - v2) * (v1 - v2);
    }
    
    return sum;
  }


  size_t size() const {
    DBG(std::cerr << "Size called " << std::endl);
    DBG(std::cerr << "size _object refcount: " << _object->ob_refcnt << std::endl);
    if (_object==Py_None) {
      DBG(std::cerr << "Size return 0" << std::endl);
      return 0;
    }
    size_t s = PyObject_Length(_object);
    DBG(std::cerr << "Size return " << s << std::endl);
    DBG(std::cerr << "size _object refcount: " << _object->ob_refcnt << std::endl);
    return s;
  }

  size_t loaded() const {
    return size();
  }

  size_t dim() const {
    return _d;
  }

 protected:  
  long            _d;
  PyObject      * _object;
  PyArrayObject * _array;

  void set_dim() {
    _d = 0;
    if (_object==Py_None) return;

    DBG(std::cerr << "Getting shape" << std::endl);
    PyObject * shape = PyObject_GetAttrString(_object, "shape");
    DBG(std::cerr << "Got shape, getting dim" << std::endl);
    if (PyTuple_Size(shape) != 2) {
      return;
    }
    PyObject * dim = PyTuple_GetItem(shape, 1);
    DBG(std::cerr << "Got dim" << std::endl);
    _d = 0;
    if (dim == nullptr) {
      DBG(std::cerr << "dim is null" << std::endl);
    }
    else if (PyLong_Check(dim)) {
      _d = PyLong_AsLong(dim);
      DBG(std::cerr << "dim is a long" << std::endl);
    }
    else if (PyInt_Check(dim)) {
      _d = PyInt_AsLong(dim);
      DBG(std::cerr << "dim is an int" << std::endl);
    }
    else {
      DBG(std::cerr << "dim is not a number" << std::endl);
    }
    DBG(std::cerr << "dim is: " << _d << std::endl);
    Py_DECREF(shape);
    DBG(std::cerr << "set_dim _object refcount: " << _object->ob_refcnt << std::endl);
  }
};

// TODO
class PyDataSink
{
public:
  typedef size_t IDType;
  typedef float DistanceType;
  typedef L2<float> Distance;

 PyDataSink(PyObject * neighbors, PyObject * distances)
   : _neighbors(neighbors), _distances(distances),
     _aneighbors(nullptr), _adistances(nullptr), 
     _d(0) {
   Py_INCREF(_neighbors);
   Py_INCREF(_distances);
   import_array_wrap();
   if(PyArray_Check(_neighbors)) {
     PyArrayObject * array = (PyArrayObject*)_neighbors;
     if (PyArray_IS_C_CONTIGUOUS(array)
         && (PyArray_TYPE(array) == NPY_INT)) {
       _aneighbors = array;
     }
     if (*PyArray_SHAPE(array) != 2) {
       //PyErr_SetString(PyExc_ValueError, "Neighbors should be a 2-dim object");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Neighbors should be a 2-dim object"); //generates a ValueError

     }
     _d = PyArray_DIM(array, 1);
   }
   else {
     PyObject * shape = PyObject_GetAttrString(_neighbors, "shape");
     if (PyTuple_Size(shape) != 2) {
       Py_DECREF(shape);
       //PyErr_SetString(PyExc_ValueError, "Neighbors should be a 2-dim object");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Neighbors should be a 2-dim object");
     }
     PyObject * dim = PyTuple_GetItem(shape, 1);
     if (dim == nullptr) {
       Py_DECREF(shape);
       //PyErr_SetString(PyExc_ValueError, "Neighbors should have a valid 1st axis");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Neighbors should have a valid 1st axis");
     }
     else if (PyLong_Check(dim)) {
       _d = PyLong_AsLong(dim);
     }
     else if (PyInt_Check(dim)) {
       _d = PyInt_AsLong(dim);
     }
     else {
       Py_DECREF(dim);
       Py_DECREF(shape);
       //PyErr_SetString(PyExc_ValueError, "Neighbors dimension is not a known number type");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Neighbors dimension is not a known number type");
     }
     DBG(std::cerr << "dim is: " << _d << std::endl);
     Py_DECREF(dim);
     Py_DECREF(shape);
   }
   if(PyArray_Check(_distances)) {
     PyArrayObject * array = (PyArrayObject*)_distances;
     if (PyArray_IS_C_CONTIGUOUS(array)
         && (PyArray_TYPE(array) == NPY_FLOAT)) {
       _adistances = array;
     }
     if (*PyArray_SHAPE(array) != 2) {
       //PyErr_SetString(PyExc_ValueError, "Distances should be a 2-dim object");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Distances should be a 2-dim object");
     }
     if (_d != PyArray_DIM(array, 1)) {
       //PyErr_SetString(PyExc_ValueError, "Distances dimension should be the same as Neighbors");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Distances dimension should be the same as Neighbors");
     }
   }
   else {
     PyObject * shape = PyObject_GetAttrString(_distances, "shape");
     if (PyTuple_Size(shape) != 2) {
       Py_DECREF(shape);
       //PyErr_SetString(PyExc_ValueError, "Distances should be a 2-dim object");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Distances should be a 2-dim object");
     }
     PyObject * dim = PyTuple_GetItem(shape, 1);
     long d = 0;
     if (dim == nullptr) {
       Py_DECREF(shape);
       //PyErr_SetString(PyExc_ValueError, "Distances should have a valid 1st axis");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Distances should have a valid 1st axis");
     }
     else if (PyLong_Check(dim)) {
       d = PyLong_AsLong(dim);
     }
     else if (PyInt_Check(dim)) {
       d = PyInt_AsLong(dim);
     }
     else {
       Py_DECREF(dim);
       Py_DECREF(shape);
       //PyErr_SetString(PyExc_ValueError, "Distances dimension is not a known number type");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Distances dimension is not a known number type");
     }
     Py_DECREF(dim);
     Py_DECREF(shape);
     if (_d != d) {
       //PyErr_SetString(PyExc_ValueError, "Distances dimension should be the same as Neighbors");
       //return (PyObject *)NULL;
       throw std::invalid_argument("Distances dimension should be the same as Neighbors");
     }
   }

 }


#if PY_VERSION_HEX >= 0x03000000
  void* import_array_wrap()
#else
  void  import_array_wrap()
#endif
  {
    import_array(); // required to avoid core dumps from numpy
    // I wrote a wrapper for the macro import_array() since it 'returns' when there's an error but the compilers do not allow the use of a 'return' keyword in a class constructor. 
  }

  ~PyDataSink() {
    if (_distances != nullptr) {
      Py_DECREF(_distances);
    }
    _distances = nullptr;
    _adistances = nullptr;
    if (_neighbors != nullptr) {
      Py_DECREF(_neighbors);
    }
    _neighbors = nullptr;
    _aneighbors = nullptr;
  }

  const IDType * getNeighbors(IDType id) const {
    return NULL; // TODO
  }

  const DistanceType * getDistances(IDType id) const {
    return NULL; // TODO
  }

  void setNeighbors(IDType id, const IDType * neighbors_, const DistanceType * distances_) {
    // we "copy" the neighbors and distances 
    for(size_t i = 0; i < _d; ++i) {
      ; //TODO
    }
  }

 protected:  
  PyObject      * _neighbors;
  PyObject      * _distances;
  PyArrayObject * _aneighbors;
  PyArrayObject * _adistances;
  size_t          _d;
};

typedef Neighbor<size_t, float> PyNeighbor;

typedef ResultSet<size_t, float> PyResultSet;
typedef std::vector<ResultSet<size_t, float>> PyResultSets;
typedef std::vector<float> Point;
typedef std::vector<Point> Points;

class PyIndexL2 : public ProgressiveKDTreeIndex<PyDataSource> {
public:
  PyIndexL2(IndexParams indexParams_,
            TreeWeight weight_ = TreeWeight(0.3, 0.7),
            const float reconstructionWeight_ = .25f)
    : ProgressiveKDTreeIndex<PyDataSource>(indexParams_, weight_, reconstructionWeight_) { }
};

typedef ProgressiveKNNTable<PyIndexL2, PyDataSink> PyKNNTable;
