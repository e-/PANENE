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
        PyIndexL2(IndexParams ip, TreeWeight, float)
        void setDataSource(PyDataSource  * ds)
        size_t addPoints(size_t end)
        void beginUpdate()
        UpdateResult2 run(size_t ops)
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()
        void knnSearch(size_t id, PyResultSet& results, size_t knn, const SearchParams& params)
        void knnSearchVec(const Points& vec, PyResultSets& results, size_t knn, const SearchParams& params)

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


    cdef cppclass PyDataSink:
        PyDataSink(object neighbors, object distances) except +ValueError

    cdef cppclass PyKNNTable:
        PyKNNTable(size_t knn, size_t d, IndexParams ip, SearchParams sp, TreeWeight treew, TableWeight tablew, PyDataSink * sink)
        void setDataSource(PyDataSource * ds)
        size_t getSize()
        UpdateResult run(size_t ops)
        PyResultSet& getNeighbors(int id)
