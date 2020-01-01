from cpython cimport PyObject, Py_INCREF
from cython.operator import dereference

import warnings
import sys
import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef char* version = '0.0.1'
cdef object __version__ = version

cdef inline check_array(arr):
    shape = arr.shape
    if len(shape)!=2: # add more tests
        raise TypeError('value is %s not an array', arr)
    if len(arr) != shape[0]:
        raise TypeError('Inconsistency in array len', arr)

cdef class Index:
    cdef PyDataSource * c_src
    cdef IndexParams    c_indexParams
    cdef PyIndexL2    * c_index

    def __cinit__(self, array, w=(0.3, 0.7), float reconstruction_weight=0.25, trees = None):
        check_array(array)
        self.c_src = new PyDataSource(array)
        
        if trees is not None:
            self.c_indexParams.trees = trees

        self.c_index = new PyIndexL2(self.c_src,
                                     self.c_indexParams, TreeWeight(w[0], w[1]),
                                     reconstruction_weight)

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

    @property
    def is_using_pyarray(self):
        return self.c_src.is_using_pyarray()

    def add_points(self, size_t end):
        self.c_index.addPoints(end)

    def size(self):
        return self.c_index.getSize()

    def knn_search(self, int pid, size_t k, checks=None, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()

        if checks is not None:
            params.checks = checks
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if not self.is_using_pyarray:
            if cores is not None and cores != 1:
                warnings.warn('Ignoring cores for non-numpy arrays')
            cores = 1
        if cores is not None:
            params.cores = cores
        
        if self.c_index.getSize() < k:
            raise ValueError('k is larger than the number of points in the index. Make sure you called add_points()')

        cdef PyResultSet res = PyResultSet(k)
        with nogil:
            self.c_index.knnSearch(pid, res, k, params)

        ids = np.ndarray((1, res.k), dtype=np.int)
        dists = np.ndarray((1, res.k), dtype=np.float)

        for i in range(k):
            nei = res[i]
            ids[0, i] = nei.id
            dists[0, i] = nei.dist

        return ids, dists

    def knn_search_points(self, np.ndarray[DTYPE_t, ndim=2] points, size_t k, checks=None, eps=None, sorted=None, cores=None):
        cdef SearchParams params = SearchParams()
        if checks is not None:
            params.checks = checks
        if eps is not None:
            params.eps = eps
        if sorted is not None:
            params.sorted = sorted
        if not self.is_using_pyarray:
            if cores is not None and cores != 1:
                warnings.warn('Ignoring cores for non-numpy arrays')
                cores = 1
        if cores is not None:
            params.cores = cores

        if self.c_index.getSize() < k:
            raise ValueError('k is larger than the number of points in the index. Make sure you called add_points()')

        cdef size_t n = points.shape[0] # of query points
        cdef size_t d = points.shape[1] # dimension
        cdef Points cpoints = Points()

        cpoints.reserve(n)

        for j in range(n):
            cpoints.emplace_back(d)
            for i in range(d):
                cpoints[j][i] = points[j, i]

        cdef PyResultSets ress = PyResultSets(n)

        with nogil:
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
        with nogil:
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
    cdef PyDataSink   * c_sink
    cdef PyKNNTable   * c_table
    
    def __cinit__(self, object array, int k, object neighbors, object distances,
                  treew=(0.3, 0.7), tablew=(0.5, 0.5),
                  trees=None,
                  checks=None, eps=None, sorted=None, cores=None
                  ):
        check_array(array)
        check_array(neighbors)
        check_array(distances)
        if neighbors.shape[1] != k or distances.shape[1] != k:
            raise ValueError('neighbors and distances should have axis=1 of %d'%k)

        if trees is not None:
            self.c_indexParams.trees = trees

        if checks is not None:
            self.c_searchParams.checks = checks
        if eps is not None:
            self.c_searchParams.eps = eps
        if sorted is not None:
            self.c_searchParams.sorted = sorted
        self.c_src = new PyDataSource(array)
        self.c_sink = new PyDataSink(neighbors, distances)
        if not (self.is_using_pyarray and \
                self.is_using_neighbors_pyarray and \
                self.is_using_distances_pyarray):
            if cores is not None and cores != 1:
                warnings.warn('Ignoring cores for non-numpy arrays')
            cores = 1
        if cores is not None:
            self.c_searchParams.cores = cores

        self.c_table = new PyKNNTable(self.c_src,
                                      self.c_sink,
                                      k, 
                                      self.c_indexParams,
                                      self.c_searchParams,
                                      TreeWeight(treew[0], treew[1]),
                                      TableWeight(tablew[0], tablew[1])
                                     )

    @property
    def is_using_pyarray(self):
        return self.c_src.is_using_pyarray()

    @property
    def is_using_neighbors_pyarray(self):
        return self.c_sink.is_using_neighbors_pyarray()

    @property
    def is_using_distances_pyarray(self):
        return self.c_sink.is_using_distances_pyarray()

    def __dealloc(self):
        del self.c_table
        del self.c_src
        del self.c_sink

    def size(self):
        return self.c_table.getSize()

    def run(self, size_t ops):
        cdef UpdateResult ur
        with nogil:
            ur = self.c_table.run(ops)
        return {
            'addPointOps': ur.addPointOps,
            'updateIndexOps': ur.updateIndexOps,
            'updateTableOps': ur.updateTableOps,
            'addPointResult': ur.addPointResult,
            'updateIndexResult': ur.updateIndexResult,
            'updateTableResult': ur.updateTableResult,
            'numPointsInserted': ur.numPointsInserted,  
            'addPointElapsed': ur.addPointElapsed,
            'updateIndexElapsed': ur.updateIndexElapsed,
            'updateTableElapsed': ur.updateTableElapsed,
            'updatedIds': ur.updatedIds,
            }
