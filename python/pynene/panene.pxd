from cpython cimport PyObject
from libcpp.string cimport string
from numpy cimport int64_t, int32_t, uint32_t, float64_t

#cdef extern from "panene_python.h" namespace "panene":
#    cdef cppclass PythonDataSource:
#        PythonDataSource()
#        ~PythonDataSource()
