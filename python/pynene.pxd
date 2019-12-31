from cpython cimport PyObject
from libcpp cimport bool
from libcpp.set cimport set

from numpy cimport int64_t, int32_t, uint32_t, float64_t
cimport numpy as np

ctypedef unsigned long size_t

cdef extern from "panene_python.h":
    cdef cppclass PyDataSource:
        PyDataSource(object array)
        void set_array(object array)
        object get_array() const
        bool is_using_pyarray() const

    cdef cppclass IndexParams:
        IndexParams()
        int trees

    cdef cppclass SearchParams:
        SearchParams()
        int checks
        float eps
        int sorted
        int cores
   
    cdef cppclass TreeWeight:
        TreeWeight(float, float)

        float addPointWeight
        float updateIndexWeight

    cdef cppclass UpdateResult2:
        UpdateResult2()
        int numPointsInserted
        int addPointOps
        int updateIndexOps
        int addPointResult
        int updateIndexResult
        float addPointElapsed
        float updateIndexElapsed

    cdef cppclass PyNeighbor:
        PyNeighbor()
        int id
        float dist

    cdef cppclass PyResultSet:
        PyResultSet()
        PyResultSet(size_t)
        PyNeighbor operator[](size_t) const
        bint full() const
        size_t k
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
        void reserve(size_t)
        Point& operator[](size_t)
        size_t size() const
        void emplace_back(size_t)

    cdef cppclass PyIndexL2:
        PyIndexL2(PyDataSource *ds, IndexParams ip, TreeWeight, float)
        size_t addPoints(size_t end)
        void beginUpdate()
        UpdateResult2 run(size_t ops) nogil
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()
        void knnSearch(size_t id, PyResultSet& results, size_t knn, const SearchParams& params) nogil
        void knnSearchVec(const Points& vec, PyResultSets& results, size_t knn, const SearchParams& params) nogil

    cdef cppclass TableWeight:
        TableWeight(float treew, float tablew)
        float treeWeight
        float tableWeight

    cdef cppclass UpdateResult:
        size_t addPointOps
        size_t updateIndexOps
        size_t updateTableOps
        size_t addPointResult
        size_t updateIndexResult
        size_t updateTableResult
        size_t numPointsInserted
        double addPointElapsed
        double updateIndexElapsed
        double updateTableElapsed
        set[size_t] updatedIds


    cdef cppclass PyDataSink:
        PyDataSink(object neighbors, object distances) except +ValueError
        bool is_using_neighbors_pyarray() const
        bool is_using_distances_pyarray() const

    cdef cppclass PyKNNTable:
        PyKNNTable(PyDataSource *ds, PyDataSink *sink, size_t knn, IndexParams ip, SearchParams sp, TreeWeight treew, TableWeight tablew)
        size_t getSize()
        UpdateResult run(size_t ops) nogil
        PyResultSet& getNeighbors(int id)
