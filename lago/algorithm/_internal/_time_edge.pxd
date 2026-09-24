# Cython declarations for _time_edge.py (pure-Python mode). See _leaf.pxd.
#
# weight and duration stay Python objects: unit weights are ints and streams may
# carry floats, and the arithmetic downstream must see exactly the same types.

from lago.algorithm._internal._leaf cimport Leaf


cdef class TimeEdge:
    cdef public Leaf target
    cdef public object weight
    cdef public object duration
