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

Memory safety. Cython's per-access bounds checks are off, so every index this
function dereferences is validated explicitly instead, where it enters: array
lengths once on entry, each row's community and edge range once per row, and
each edge target once per edge. Input that would read or write outside an array
raises ``IndexError`` -- as the reference kernel does on the same input -- rather
than corrupting memory. ``lago.accel.count_intra_and_switches`` is public, so
this is not only about inputs ``lago`` builds itself. One difference from the
reference: a negative edge target raises here, where Python's negative indexing
would wrap. Cost: about 10 % of the kernel's own time, and the kernel is a small
part of a warm ``longitudinal_modularity`` call -- not measurable there.
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

    Raises:
        ValueError: ``n`` or ``n_communities`` is negative.
        IndexError: an array is shorter than ``n`` says, or a label, an edge
            range, an edge target or a left/right row points outside the arrays.
    """
    cdef Py_ssize_t row, position, start, stop
    cdef Py_ssize_t n_labels = label.shape[0]
    cdef Py_ssize_t n_edges = min(target.shape[0], weight.shape[0])
    cdef int community, other, neighbour
    cdef long long switches = 0
    cdef double total
    cdef double *intra

    if n < 0 or n_communities < 0:
        raise ValueError(f"negative size: n={n}, n_communities={n_communities}")
    if (
        n_labels < n
        or left.shape[0] < n
        or right.shape[0] < n
        or indptr.shape[0] < n + 1
    ):
        raise IndexError(f"an array is shorter than n={n}")

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
            if community >= n_communities:
                raise IndexError(f"label {community} at row {row} >= n_communities={n_communities}")

            start = indptr[row]
            stop = indptr[row + 1]
            if start < 0 or stop > n_edges:
                raise IndexError(f"edge range [{start}, {stop}) of row {row} outside the edge arrays")

            total = intra[community]
            for position in range(start, stop):
                neighbour = target[position]
                if neighbour < 0 or neighbour >= n_labels:
                    raise IndexError(f"edge target {neighbour} outside the label array")
                if label[neighbour] == community:
                    total += weight[position]
            intra[community] = total

            neighbour = left[row]
            if neighbour >= 0:
                if neighbour >= n_labels:
                    raise IndexError(f"left row {neighbour} outside the label array")
                other = label[neighbour]
                if other >= 0 and other != community:
                    switches += 1
            neighbour = right[row]
            if neighbour >= 0:
                if neighbour >= n_labels:
                    raise IndexError(f"right row {neighbour} outside the label array")
                other = label[neighbour]
                if other >= 0 and other != community:
                    switches += 1

        return [intra[community] for community in range(n_communities)], switches
    finally:
        PyMem_Free(intra)
