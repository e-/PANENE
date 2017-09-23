from cpython cimport PyObject
from libcpp.string cimport string
from numpy cimport int64_t, int32_t, uint32_t, float64_t

ctypedef unsigned long size_t

cdef extern from "panene_python.h":
    cdef cppclass PythonDataSource:
        PythonDataSource()
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

    cdef cppclass IndexL2:
        IndexL2(IndexParams ip)
        void setDataSource(PythonDataSource  * ds)
        size_t addPoint(size_t end)
        size_t update(int ops)
        void removePoint(size_t id)
        size_t getSize()
        int usedMemory()

