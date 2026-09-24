# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
"""Compiled counting kernel for ``lago.accel``.

A transcription of ``lago.accel._reference_kernel``, which is the contract. It
must stay a transcription: the two loops are fused and the weights are summed in
the order the arrays are laid out, which is the order the Python loops in
``lago.metrics.modularity`` visit them, so every partial sum -- and therefore
the final double -- is bit-identical.

Deliberately absent:

* No reassociation, no vectorisation hints, no ``-ffast-math``. Any of them
  would reorder the additions and break bit-identity for non-integer weights.
* No parallelism. Accumulating per community from several threads would make
  the summation order depend on scheduling.

The one arithmetic liberty is that the accumulator is a C double where Python
uses an int until the first float weight arrives. Integer weights agree exactly
while a community's total stays below 2**53, which is 9e15 interactions.
"""

from cpython.mem cimport PyMem_Free, PyMem_Malloc


def count_intra_and_switches(
    Py_ssize_t n,
    const int[::1] label,
    const long long[::1] indptr,
    const int[::1] target,
    const double[::1] weight,
    const int[::1] left,
    const int[::1] right,
    Py_ssize_t n_communities,
):
    """Sum intra-community weight and count community switches in one pass.

    Args:
        n: Number of time-nodes.
        label: Community index per time-node, -1 for unlabelled.
        indptr: Row starts into ``target`` / ``weight``, length ``n + 1``.
        target: Neighbour row of each incident edge.
        weight: Weight of each incident edge.
        left: Previous active time-node, or -1.
        right: Next active time-node, or -1.
        n_communities: Number of distinct communities.

    Returns:
        ``(intra, switches)`` -- the per-community interaction sum as a list of
        floats, and twice the number of community switches.
    """
    cdef Py_ssize_t row, position, stop
    cdef int community, other
    cdef long long switches = 0
    cdef double total
    cdef double *intra

    if n_communities == 0:
        # Nothing is labelled, so no edge and no time step can contribute.
        return [], 0

    intra = <double *> PyMem_Malloc(n_communities * sizeof(double))
    if intra is NULL:
        raise MemoryError

    try:
        for community in range(n_communities):
            intra[community] = 0.0

        for row in range(n):
            community = label[row]
            if community < 0:
                continue

            total = intra[community]
            stop = indptr[row + 1]
            for position in range(indptr[row], stop):
                if label[target[position]] == community:
                    total += weight[position]
            intra[community] = total

            other = left[row]
            if other >= 0:
                other = label[other]
                if other >= 0 and other != community:
                    switches += 1
            other = right[row]
            if other >= 0:
                other = label[other]
                if other >= 0 and other != community:
                    switches += 1

        return [intra[community] for community in range(n_communities)], switches
    finally:
        PyMem_Free(intra)
