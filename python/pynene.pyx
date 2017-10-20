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

    def __cinit__(self, array):
        check_array(array)
        self.c_src = new PyDataSource(array)
        self.c_index = new PyIndexL2(self.c_indexParams)
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

    def knn_search(self, int val, k, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if cores is not None:
            params.cores = cores
        cdef PyResultSet res = PyResultSet(k)
        self.c_index.knnSearch(val, res, k, params)
        ids = np.ndarray((1, res.size), dtype=np.int)
        dists = np.ndarray((1, res.size), dtype=np.float)
        for i in range(res.size):
            n = res[i]
            ids[0, i] = n.id
            dists[0, i] = n.dist
        return ids, dists

#    @cython.boundscheck(False) # turn off bounds-checking for entire function
#    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def knn_search_points(self, np.ndarray[DTYPE_t, ndim=2] val, k, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if cores is not None:
            params.cores = cores
        cdef int l = val.shape[0]
        cdef int d = val.shape[1]
        cdef Points pts = Points(l)
        cdef Point p
        for j in range(l):
            p = Point(d)
            pts[j] = p
            for i in range(d):
                pts[j][d] = val[j, i]
        cdef PyResultSets ress = PyResultSets()
        self.c_index.knnSearchVec(pts, ress, k, params)
        ids = np.ndarray((ress.size(), k), dtype=np.int)
        dists = np.ndarray((ress.size(), k), dtype=np.float32)
        cdef PyResultSet res
        for j in range(ress.size()):
            res = ress[j]
            for i in range(res.size):
                n = res[i]
                ids[i] = n.id
                dists[i] = n.dist
        return ids, dists
        
