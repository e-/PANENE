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
  typedef L2<float> Distance;
  typedef float ElementType;
  typedef float DistanceType;

  PyDataSource(PyObject * o)
    : _d(0),
    _object(Py_None),
    _array(nullptr) {
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
#if PY_VERSION_HEX >= 0x03000000
    return NULL;
#endif
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
    _array = nullptr;
    Py_DECREF(_object);
    _object = o;
    DBG(std::cerr << "set_array _object refcount: " << _object->ob_refcnt << std::endl);
    if (_object != Py_None) {
      if(PyArray_Check(_object) && PyArray_ISCARRAY_RO(_object)) {
        DBG(std::cerr << "Object is a C contiguous array acceptable for fast get"  << std::endl);
        _array = (PyArrayObject*)_object;
      } 
      else {
        DBG(std::cerr << "Object is not acceptable for fast get...");
      }
    }
    else {
      DBG(std::cerr << "Object is None...");
    }

    set_dim();
  }

  PyObject * get_array() const {
    Py_INCREF(_object); //TODO check that this is the right way to do it
    return _object;
  }

  bool is_using_pyarray() const { return _array != nullptr; }

  ElementType get(const IDType &id, const IDType &dim) const {
    if (_array != nullptr) {
      //DBG(std::cerr << "get from array" << std::endl);
      void * ptr = PyArray_GETPTR2(_array, id, dim);
      switch (PyArray_TYPE(_array)) {
      case NPY_FLOAT: return *(npy_float *) ptr;
      case NPY_DOUBLE: return (ElementType)*(npy_double*)ptr;
      case NPY_LONGDOUBLE: return (ElementType)*(npy_longdouble*)ptr;
      case NPY_BYTE: return (ElementType)*(npy_byte *)ptr;
      case NPY_UBYTE: return (ElementType)*(npy_ubyte *)ptr;
      case NPY_SHORT: return (ElementType)*(npy_short *)ptr;
      case NPY_USHORT: return (ElementType)*(npy_ushort *)ptr;
      case NPY_INT: return (ElementType)*(npy_int *)ptr;
      case NPY_UINT: return (ElementType)*(npy_uint *)ptr;
      case NPY_LONG: return (ElementType)*(npy_long *)ptr;
      case NPY_ULONG: return (ElementType)*(npy_ulong *)ptr;
      case NPY_LONGLONG: return (ElementType)*(npy_longlong *)ptr;
      case NPY_ULONGLONG: return (ElementType)*(npy_ulonglong *)ptr;
      }
      // Fall through
    }
    DBG(std::cerr << "Starting get(" << id << "," << dim << ")" << std::endl);
    PyObject *tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, PyInt_FromLong(id)); // steals reference 
    PyTuple_SetItem(tuple, 1, PyInt_FromLong(dim));
    PyObject *item = PyObject_GetItem(_object, tuple);
    ElementType ret = NPY_NAN;
    DBG(std::cerr << "Got item " << item);
    if (item != nullptr) {
      DBG(std::cerr << " type: " << item->ob_type->tp_name << std::endl);
      ret = (ElementType)PyFloat_AsDouble(item);
      Py_DECREF(item);
    }
    else {
      DBG(std::cerr << " not a valid item " << std::endl);
    }
    DBG(std::cerr << " value is: " << ret << std::endl);
    Py_DECREF(tuple);
    return ret;
  }

  void get(const IDType &id, std::vector<ElementType> &result) const {
    size_t d = dim();
    if (_array != nullptr) {
      switch (PyArray_TYPE(_array)) {
      case NPY_FLOAT: {
        float * ptr = (float *)PyArray_GETPTR2(_array, id, 0);
        for(size_t i=0;i<d;++i) result[i] = ptr[i]; }
        return;
#define CASE(TYPE,type)                                         \
      case TYPE: {                                              \
        type * ptr = (type *)PyArray_GETPTR2(_array, id, 0);    \
        for(size_t i=0;i<d;++i) result[i] = (float)ptr[i]; }    \
        return
        CASE(NPY_DOUBLE, npy_double);
        CASE(NPY_LONGDOUBLE, npy_longdouble);
        CASE(NPY_BYTE, npy_byte);
        CASE(NPY_UBYTE, npy_ubyte);
        CASE(NPY_SHORT, npy_short);
        CASE(NPY_USHORT, npy_ushort);
        CASE(NPY_INT, npy_int);
        CASE(NPY_UINT, npy_uint);
        CASE(NPY_LONG, npy_long);
        CASE(NPY_ULONG, npy_ulong);
        CASE(NPY_LONGLONG, npy_longlong);
        CASE(NPY_ULONGLONG, npy_ulonglong);
#undef CASE
      }
      // Fall through
    }
    for(unsigned int i=0;i < d;++i) {
      result[i] = get(id, i);
    }
  }

  IDType findDimWithMaxSpan(const IDType &id1, const IDType &id2) {
    size_t dimension = 0;
    ElementType maxSpan = 0;
    size_t d = dim();

    std::vector<ElementType> row1(d);
    get(id1, row1);
    std::vector<ElementType> row2(d);
    get(id2, row2);

    for(size_t i = 0; i < d; ++i) {
      ElementType span = std::abs(row1[i] - row2[i]);
      if(maxSpan < span) {
        maxSpan = span;
        dimension = i;
      }
    }

    return dimension;
  }

  void computeMeanAndVar(const IDType *ids, int count, std::vector<DistanceType> &mean, std::vector<DistanceType> &var) {
    size_t d = dim();
    std::vector<ElementType> row(d);

    mean.resize(d);
    var.resize(d);

    for (size_t i = 0; i < d; ++i) 
      mean[i] = var[i] = 0;

    
    for (int j = 0; j < count; ++j) {
      get(ids[j], row);

      for (size_t i = 0; i < d; ++i) {
        mean[i] += row[i];
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
    std::vector<ElementType> row1(d);
    get(id1, row1);
    std::vector<ElementType> row2(d);
    get(id2, row2);


    for(size_t i = 0; i < d; ++i) {
      ElementType v1 = row1[i], v2 = row2[i];
      sum += (v1 - v2) * (v1 - v2);
    }
    
    return sum;
  }

  DistanceType getSquaredDistance(const IDType &id1, const std::vector<ElementType> &vec2) const {
    DistanceType sum = 0;
    size_t d = dim();
    std::vector<ElementType> row1(d);
    get(id1, row1);

    for(size_t i = 0; i < d; ++i) {
      ElementType v1 = row1[i], v2 = vec2[i];
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

  size_t capacity() const {
    return size();
  }

  size_t dim() const {
    return _d;
  }

 protected:  
  long            _d;
  PyObject      * _object;
  PyArrayObject * _array;
  bool            _is_float;

  void set_dim() {
    _d = 0;
    if (_object==Py_None) return;

    if (_array != nullptr) {
      _d = PyArray_DIM(_array, 1);
      DBG(std::cerr << "Fast set_dim is: " << _d << std::endl);
      return;
    }

    DBG(std::cerr << "Getting shape" << std::endl);
    PyObject * shape = PyObject_GetAttrString(_object, "shape");
    DBG(std::cerr << "Got shape, getting dim" << std::endl);
    if (PyTuple_Size(shape) != 2) {
      throw std::invalid_argument("Array should be a 2-dim object"); //generates a ValueError
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

class PyDataSink
{
public:
  typedef size_t IDType;
  typedef float DistanceType;
  typedef L2<float> Distance;

 PyDataSink(PyObject * neighbors, PyObject * distances)
   : _neighbors(neighbors), _distances(distances),
     _aneighbors(nullptr), _adistances(nullptr), 
     _d(0),
     _distance_cache(nullptr), _neighbor_cache(nullptr),
     _last_distance_id(-1), _last_neighbor_id(-1) {
   Py_INCREF(_neighbors);
   Py_INCREF(_distances);
   import_array_wrap();
   if (PyArray_Check(_neighbors)
       && PyArray_ISCARRAY_RO(_neighbors)
       && (PyArray_ISINTEGER(_neighbors))) {
     _aneighbors = (PyArrayObject*)_neighbors;
     DBG(std::cerr << "PyDataSink neighbors is an acceptable array" << std::endl);
     if (PyArray_NDIM(_aneighbors) != 2) {
       throw std::invalid_argument("Neighbors should be a 2-dim object"); //generates a ValueError
     }
     _d = PyArray_DIM(_aneighbors, 1);
   }
   else {
     DBG(std::cerr << "PyDataSink neigbbors is NOT an acceptable array" << std::endl);
     PyObject * shape = PyObject_GetAttrString(_neighbors, "shape");
     if (PyTuple_Size(shape) != 2) {
       Py_DECREF(shape);
       throw std::invalid_argument("Neighbors should be a 2-dim object");
     }
     PyObject * dim = PyTuple_GetItem(shape, 1);
     if (dim == nullptr) {
       Py_DECREF(shape);
       throw std::invalid_argument("Neighbors should have a valid 1st axis");
     }
     else if (PyLong_Check(dim)) {
       _d = PyLong_AsLong(dim);
     }
     else if (PyInt_Check(dim)) {
       _d = PyInt_AsLong(dim);
     }
     else {
       //Py_DECREF(dim); PyTuple_GetItem returns a borrowed ref, no decref needed
       Py_DECREF(shape);
       throw std::invalid_argument("Neighbors dimension is not a known number type");
     }
     // Py_DECREF(dim); PyTuple_GetItem returns a borrowed ref, no decref needed
     Py_DECREF(shape);
     _neighbor_cache = new IDType[_d];
   }
   DBG(std::cerr << "dim is: " << _d << std::endl);

   if (PyArray_Check(_distances)
       && PyArray_ISCARRAY_RO(_distances)
       && (PyArray_TYPE(_distances)==NPY_FLOAT || PyArray_TYPE(_distances)==NPY_DOUBLE)) {
     DBG(std::cerr << "PyDataSink distances is an acceptable array" << std::endl);
     _adistances = (PyArrayObject*)_distances;
     if (PyArray_NDIM(_adistances) != 2) {
       throw std::invalid_argument("Distances should be a 2-dim object");
     }
     if (_d != PyArray_DIM(_adistances, 1)) {
       throw std::invalid_argument("Distances dimension should be the same as Neighbors");
     }
   }
   else {
     DBG(std::cerr << "PyDataSink neigbbors is NOT an acceptable array" << std::endl);
     PyObject * shape = PyObject_GetAttrString(_distances, "shape");
     if (PyTuple_Size(shape) != 2) {
       Py_DECREF(shape);
       throw std::invalid_argument("Distances should be a 2-dim object");
     }
     PyObject * dim = PyTuple_GetItem(shape, 1);
     long d = 0;
     if (dim == nullptr) {
       Py_DECREF(shape);
       throw std::invalid_argument("Distances should have a valid 1st axis");
     }
     else if (PyLong_Check(dim)) {
       d = PyLong_AsLong(dim);
     }
     else if (PyInt_Check(dim)) {
       d = PyInt_AsLong(dim);
     }
     else {
       // Py_DECREF(dim); PyTuple_GetItem returns a borrowed ref, no decref needed
       Py_DECREF(shape);
       throw std::invalid_argument("Distances dimension is not a known number type");
     }
     // Py_DECREF(dim); PyTuple_GetItem returns a borrowed ref, no decref needed
     Py_DECREF(shape);
     if (_d != d) {
       throw std::invalid_argument("Distances dimension should be the same as Neighbors");
     }
     _distance_cache = new DistanceType[_d];
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
#if PY_VERSION_HEX >= 0x03000000
    return NULL;
#endif
  }

  ~PyDataSink() {
    DBG(std::cerr << "PyDataSink calling destructor" << std::endl);
    _adistances = nullptr;
    if (_distances != nullptr) {
      Py_DECREF(_distances);
    }
    _distances = nullptr;
    _aneighbors = nullptr;
    if (_neighbors != nullptr) {
      Py_DECREF(_neighbors);
    }
    _neighbors = nullptr;
    if (_distance_cache != nullptr) {
      delete _distance_cache;
      _distance_cache = nullptr;
    }
    if (_neighbor_cache != nullptr) {
      delete _neighbor_cache;
      _neighbor_cache = nullptr;
    }
  }

  bool is_using_neighbors_pyarray() const { return _aneighbors != nullptr; }
  bool is_using_distances_pyarray() const { return _adistances != nullptr; }

  // TODO: now we can remove neighbor cache
  void getNeighbors(const IDType id, std::vector<IDType>& res) const {
    DBG(std::cerr << "PyDataSink getNeighbors(" << id << ")" << std::endl);
    if (_aneighbors != nullptr) {
      void * ptr = PyArray_GETPTR2(_aneighbors, id, 0);
      switch (PyArray_TYPE(_aneighbors)) {
#define CASE(TYPE, type) \
      case TYPE: { type * begin = (type *)ptr; res.assign(begin, begin+_d); } return
      CASE(NPY_BYTE, npy_byte);
      CASE(NPY_UBYTE, npy_ubyte);
      CASE(NPY_SHORT, npy_short);
      CASE(NPY_USHORT, npy_ushort);
      CASE(NPY_INT, npy_int);
      CASE(NPY_UINT, npy_uint);
      CASE(NPY_LONG, npy_long);
      CASE(NPY_ULONG, npy_ulong);
      CASE(NPY_LONGLONG, npy_longlong);
      CASE(NPY_ULONGLONG, npy_ulonglong);
#undef CASE        
      // Fall through
      }
    }
    std::vector<IDType> ret(_d);
    if (_last_neighbor_id != id) {
      _last_neighbor_id = id;
      IDType v;
      PyObject *tuple = PyTuple_New(2);
      PyTuple_SetItem(tuple, 0, PyInt_FromLong(id));
      PyTuple_SetItem(tuple, 0, PyInt_FromLong(0));
      PyObject *item = PyObject_GetItem(_neighbors, tuple);
      v = 0;
      if (PyLong_Check(item)) {
        v = PyLong_AsLong(item);
      }
      else if (PyInt_Check(item)) {
        v = PyInt_AsLong(item);
      }
      if (item != nullptr) {
        Py_DECREF(item);
      }
      _neighbor_cache[0] = v;
      for(int i = 1; i < _d; ++i) {
        PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(i));
        item = PyObject_GetItem(_neighbors, tuple);
        v = 0;
        if (PyLong_Check(item)) {
          v = PyLong_AsLong(item);
        }
        else if (PyInt_Check(item)) {
          v = PyInt_AsLong(item);
        }
        if (item != nullptr) {
          Py_DECREF(item);
        }
        _neighbor_cache[i] = v;
      }
      Py_DECREF(tuple);
    }
    res.assign(_neighbor_cache, _neighbor_cache+_d);
  }

  // TODO: now we can remove distance cache
  void getDistances(const IDType id, std::vector<DistanceType>& res) const {
    DBG(std::cerr << "PyDataSink getDistances(" << id << ")" << std::endl);
    if (_adistances != nullptr) {
      //DistanceType * begin = (DistanceType *)PyArray_GETPTR2(_adistances, id, 0);
      void * ptr = PyArray_GETPTR2(_adistances, id, 0);
      switch (PyArray_TYPE(_aneighbors)) {
#define CASE(TYPE, type)                                                \
        case TYPE: { type * begin = (type *)ptr; res.assign(begin, begin+_d); } return
        CASE(NPY_FLOAT, npy_float);
        CASE(NPY_DOUBLE, npy_double);
        CASE(NPY_LONGDOUBLE, npy_longdouble);
#undef CASE        
        // Fall through
      }
    }
    _last_distance_id = id;
    DistanceType v;
    PyObject *tuple = PyTuple_New(2);
    PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong(id));
    PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(0));
    PyObject *item = PyObject_GetItem(_distances, tuple);
    v = 0;
    if (PyFloat_Check(item)) {
      v = (DistanceType)PyFloat_AsDouble(item);
    }
    if (item != nullptr) {
      Py_DECREF(item);
    }
    _distance_cache[0] = v;
    for(int i = 1; i < _d; ++i) {
      PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(i));
      item = PyObject_GetItem(_distances, tuple);
      v = 0;
      if (PyFloat_Check(item)) {
        v = (DistanceType)PyFloat_AsDouble(item);
      }
      if (item != nullptr) {
        Py_DECREF(item);
      }
      _distance_cache[i] = v;
    }
    Py_DECREF(tuple);
    res.assign(_distance_cache, _distance_cache+_d);
  }


  void setNeighbors(IDType id, const IDType * neighbors_, const DistanceType * distances_) {
    int i;
    bool done = false;
    DBG(std::cerr << "PyDataSink setNeighbors(" << id << ")" << std::endl);

    if (_aneighbors != nullptr) {
      void * ptr = PyArray_GETPTR2(_aneighbors, id, 0);
      done = true;
#define CASE(TYPE, type)                                                \
      case TYPE: {                                                      \
        type * head = (type *)ptr;                                      \
        for(int i = 0; i < _d; ++i)                                     \
          head[i] = (type)neighbors_[i]; }; break
      switch (PyArray_TYPE(_aneighbors)) {
        CASE(NPY_BYTE, npy_byte);
        CASE(NPY_UBYTE, npy_ubyte);
        CASE(NPY_SHORT, npy_short);
        CASE(NPY_USHORT, npy_ushort);
        CASE(NPY_INT, npy_int);
        CASE(NPY_UINT, npy_uint);
        CASE(NPY_LONG, npy_long);
        CASE(NPY_ULONG, npy_ulong);
        CASE(NPY_LONGLONG, npy_longlong);
        CASE(NPY_ULONGLONG, npy_ulonglong);
      default: done=false;
      }
    }
#undef CASE
    if (! done) {
      _last_neighbor_id = id;
      for (i = 0; i < _d; i++) {
        _neighbor_cache[i] = neighbors_[i];
      }
      PyObject *tuple = PyTuple_New(2);
      PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong(id));
      PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(0));
      PyObject *v = PyInt_FromLong(_neighbor_cache[0]);
      if (PyObject_SetItem(_neighbors, tuple, v)==-1) {
        Py_DECREF(v);
        Py_DECREF(tuple);
        throw std::invalid_argument("setitem failed on neighbors");
      }
      for(i = 1; i < _d; ++i) { // Reuse the tuple, is it bad?
        PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(i));
        Py_DECREF(v); 
        v = PyInt_FromLong(_neighbor_cache[i]);
        if (PyObject_SetItem(_neighbors, tuple, v)==-1) {
          Py_DECREF(v);
          Py_DECREF(tuple);
          throw std::invalid_argument("setitem failed on neighbors");
        }
      }
      Py_DECREF(v);
      Py_DECREF(tuple);
    }

    done = false;
    if (_adistances != nullptr) {
      void * ptr = PyArray_GETPTR2(_adistances, id, 0);
      done = true;
#define CASE(TYPE, type)                                                \
      case TYPE: {                                                      \
        type * head = (type *)ptr;                                      \
        for (int i = 0; i < _d; ++i)                                    \
          head[i] = (type)distances_[i]; };                             \
        break
      switch (PyArray_TYPE(_adistances)) {
        CASE(NPY_FLOAT, npy_float);
        CASE(NPY_DOUBLE, npy_double);
        CASE(NPY_LONGDOUBLE, npy_longdouble);
      default: done=false;
      }
    }
#undef CASE
    if (! done) {
      _last_distance_id = id;
      for (i = 0; i < _d; i++) {
        _distance_cache[i] = distances_[i];
      }
      PyObject *tuple = PyTuple_New(2);
      PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong(id));
      PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(0));
      PyObject *v = PyFloat_FromDouble(_distance_cache[0]);
      if (PyObject_SetItem(_distances, tuple, v)==-1) {
        Py_DECREF(v);
        Py_DECREF(tuple);
        throw std::invalid_argument("setitem failed on distances");
      }
      for(int i = 1; i < _d; ++i) {
        PyObject *key2 = PyInt_FromLong(i);
        PyTuple_SET_ITEM(tuple, 1, key2);
        Py_DECREF(v);
        PyObject *v = PyFloat_FromDouble(_distance_cache[i]);
        if (PyObject_SetItem(_distances, tuple, v)==-1) {
          Py_DECREF(v);
          Py_DECREF(tuple);
          throw std::invalid_argument("setitem failed on distances");
        }
      }
      Py_DECREF(v);
      Py_DECREF(tuple);
    }
  }

 protected:  
  PyObject       * _neighbors;
  PyObject       * _distances;
  PyArrayObject  * _aneighbors;
  PyArrayObject  * _adistances;
  npy_intp         _d;
  mutable float  * _distance_cache;
  mutable IDType * _neighbor_cache;
  mutable IDType _last_distance_id;
  mutable IDType _last_neighbor_id;
};

typedef Neighbor<size_t, float> PyNeighbor;

typedef ResultSet<size_t, float> PyResultSet;
typedef std::vector<ResultSet<size_t, float>> PyResultSets;
typedef std::vector<float> Point;
typedef std::vector<Point> Points;

typedef ProgressiveKDTreeIndex<PyDataSource> PyIndexL2;
typedef ProgressiveKNNTable<PyIndexL2, PyDataSink> PyKNNTable;
