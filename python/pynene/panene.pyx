from cpython cimport PyObject, Py_INCREF
from cython.operator import dereference

from numpy cimport *

import numpy as np

cimport cython
cimport numpy as cnp

cnp.import_array()
cnp.import_ufunc()

cdef inline check_array(arr):
    if len(arr.shape)!=2: # add more tests
        raise TypeError('value is %s not an array', arr)

cdef class Index:
    cdef PyDataSource c_src
    cdef IndexParams c_indexParams
    cdef PyIndexL2 * c_index

    def __cinit__(self, array):
        self.c_src = PyDataSource()
        self.c_src.set_array(array)
        self.c_index = new PyIndexL2(self.c_indexParams)

    def __dealloc__(self):
        del self.c_index

    @property
    def array(self):
        return self.c_src.get_array()

    @array.setter
    def array(self, value):
        check_array(value)
        self.c_src.set_array(value)

    def add_points(self, size_t end):
        self.c_index.addPoints(end)

    def knn_search(self, id, k, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if cores is not None:
            params.cores = cores
        cdef PyResultSet res = PyResultSet(k)
        self.c_index.knnSearch(id, res, k, params)
        ids = np.ndarray(res.size, dtype=np.int)
        dists = np.ndarray(res.size, dtype=np.float)
        for i in range(res.size):
            n = res[i]
            ids[i] = n.id
            dists[i] = n.dist
        return ids, dists

