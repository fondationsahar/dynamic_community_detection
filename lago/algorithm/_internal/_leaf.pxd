# Cython declarations for _leaf.py (pure-Python mode).
#
# With this file beside it, `accel_cython/setup_core.py` compiles _leaf.py into
# an extension type: the attributes become C struct fields and __hash__ becomes
# a C slot. Without the build, _leaf.py runs unchanged as plain Python. Nothing
# here changes what the class does -- only how the interpreter reaches it.
#
# Why it matters: Leaf.__hash__ is called for every set and dict operation on a
# time-node -- tens of millions of times per run -- and as a Python method each
# call is an interpreter frame. Measured at 17-19 % of lago_modules.

cdef class Leaf:
    cdef public object node
    cdef public object time
    cdef public Py_hash_t _hash
    cdef public Leaf left_time_active_neighbor
    cdef public Leaf right_time_active_neighbor
    cdef public set topo_neighbors
    cdef public set topo_neighbors_from
    cdef public object module
    cdef public long edge_duration
    cdef public long _accel_row
