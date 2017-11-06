from cpython cimport PyObject, Py_INCREF
from cython.operator import dereference

import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef char* version = '0.0.1'
cdef object __version__ = version

cdef inline check_array(arr):
    if len(arr.shape)!=2: # add more tests
        raise TypeError('value is %s not an array', arr)

cdef class Index:
    cdef PyDataSource * c_src
    cdef IndexParams    c_indexParams
    cdef PyIndexL2    * c_index

    def __cinit__(self, array, weights=(0.3, 0.7), float reconstruction_weight=0.25):
        check_array(array)
        self.c_src = new PyDataSource(array)

        self.c_index = new PyIndexL2(self.c_indexParams, TreeWeight(weights[0], weights[1]), reconstruction_weight)

        self.c_index.setDataSource(self.c_src)

    def __dealloc__(self):
        del self.c_index
        del self.c_src

    @property
    def array(self):
        return self.c_src.get_array()

    @array.setter
    def array(self, value):
        check_array(value)
        self.c_src.set_array(value)

    def add_points(self, size_t end):
        self.c_index.addPoints(end)

    def size(self):
        return self.c_index.getSize()

    def knn_search(self, int pid, int k, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if cores is not None:
            params.cores = cores
        
        if self.c_index.getSize() < k:
            raise ValueError('k is larger than the number of points in the index. Make sure you called add_points()')

        cdef PyResultSet res = PyResultSet(k)
        self.c_index.knnSearch(pid, res, k, params)

        ids = np.ndarray((1, res.size), dtype=np.int)
        dists = np.ndarray((1, res.size), dtype=np.float)

        for i in range(k):
            nei = res[i]
            ids[0, i] = nei.id
            dists[0, i] = nei.dist

        return ids, dists

#    @cython.boundscheck(False) # turn off bounds-checking for entire function
#    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def knn_search_points(self, np.ndarray[DTYPE_t, ndim=2] points, int k, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if cores is not None:
            params.cores = cores

        if self.c_index.getSize() < k:
            raise ValueError('k is larger than the number of points in the index. Make sure you called add_points()')

        cdef int n = points.shape[0] # of query points
        cdef int d = points.shape[1] # dimension
        cdef Points cpoints = Points()

        cpoints.reserve(n)

        for j in range(n):
            cpoints.emplace_back(d)
            for i in range(d):
                cpoints[j][i] = points[j, i]

        cdef PyResultSets ress = PyResultSets(n)

        self.c_index.knnSearchVec(cpoints, ress, k, params)

        ids = np.ndarray((n, k), dtype=np.int)
        dists = np.ndarray((n, k), dtype=np.float32)
        cdef PyResultSet res

        for j in range(n):
            res = ress[j]
            for i in range(k):
                nei = res[i]
                ids[j][i] = nei.id
                dists[j][i] = nei.dist
                
        return ids, dists

    def run(self, int ops):
        cdef UpdateResult2 ur
        ur = self.c_index.run(ops)

        return {
            'numPointsInserted': ur.numPointsInserted,
            'addPointOps': ur.addPointOps,
            'updateIndexOps': ur.updateIndexOps,
            'addPointResult': ur.addPointResult,
            'updateIndexResult': ur.updateIndexResult,
            'addPointElapsed': ur.addPointElapsed,
            'updateIndexElapsed': ur.updateIndexElapsed
        }

cdef class KNNTable:
    cdef PyDataSource * c_src
    cdef IndexParams    c_indexParams
    cdef SearchParams   c_searchParams
    cdef PyIndexL2    * c_index

    def __cinit__(self, array, treew=(0.3, 0.7), tablew=(0.5, 0.5)):
        check_array(array)
        self.c_src = new PyDataSource(array)
        
