from cpython cimport PyObject
from numpy cimport int64_t, int32_t, uint32_t, float64_t
cimport numpy as np

ctypedef unsigned long size_t

cdef extern from "panene_python.h":
    cdef cppclass PyDataSource:
        PyDataSource(object array)
        void set_array(object array)
        object get_array() const

    cdef cppclass IndexParams:
        IndexParams()
        int trees

    cdef cppclass SearchParams:
        SearchParams()
        int checks
        float eps
        int sorted
        int cores

    cdef cppclass PyNeighbor:
        PyNeighbor()
        int id
        float dist

    cdef cppclass PyResultSet:
        PyResultSet()
        PyResultSet(size_t)
        PyNeighbor operator[](size_t) const
        bint full() const
        size_t size
        float worstDist

    cdef cppclass PyResultSets:
        PyResultSets()
        PyResultSets(size_t)
        PyResultSet operator[](size_t) const
        bint full() const
        size_t size() const

    cdef cppclass Point:
        Point()
        Point(size_t)
        float& operator[](size_t)
        size_t size() const

    cdef cppclass Points:
        Points()
        Points(size_t)
        Point& operator[](size_t)
        size_t size() const

    cdef cppclass PyIndexL2:
        PyIndexL2(IndexParams ip)
        void setDataSource(PyDataSource  * ds)
        size_t addPoints(size_t end)
        void beginUpdate()
        size_t update(int ops)
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()
        void knnSearch(size_t id, PyResultSet results, size_t knn, SearchParams params)
        void knnSearchVec(const Points& vec, PyResultSets results, size_t knn, SearchParams params)

