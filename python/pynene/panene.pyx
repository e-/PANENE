from cpython cimport PyObject, Py_INCREF

from numpy cimport *

import numpy as np

cimport cython
cimport numpy as cnp

cnp.import_array()
cnp.import_ufunc()

cdef inline check_array(arr):
    if len(arr.shape)!=2: # add more tests
        raise TypeError('value is %s not an array', arr)

cdef class PyDataSource:
    cdef PythonDataSource c_src
    def __cinit__(self, array):
        self.c_src = PythonDataSource()
        self.c_src.set_array(array)
    @property
    def array(self):
        return self.c_src.get_array()
    @array.setter
    def array(self, value):
        check_array(value)
        self.c_src.set_array(value)
            
