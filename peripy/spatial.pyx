from libc.math cimport sqrt


def euclid(r1, r2):
    """
    Calculate the Euclidean distance between two, three-dimensional
    coordinates.

    :arg r1: A (3,) array representing the first coordinate.
    :type r1: :class:`numpy.ndarray`
    :arg r2: A (3,) array representing the second coordinate.
    :type r2: :class:`numpy.ndarray`

    :returns: The Euclidean distance between `r1` and `r2`
    :rtype: `numpy.float64`
    """
    return ceuclid(r1, r2)


cdef inline float ceuclid(float[:] r1, float[:] r2):
    """
    C function for calculating the Euclidean distance between two coordinates.
    """
    cdef int imax = 3
    cdef float[3] dr

    for i in range(imax):
        dr[i] = r2[i] - r1[i]
        dr[i] = dr[i] * dr[i]

    return sqrt(dr[0] + dr[1] + dr[2])


def strain(r1, r2, r10, r20):
    """
    Calculate the strain between two particles given their current and initial
    positions.

    :arg r1: A (3,) array giving the coordinate of the first particle.
    :type r1: :class:`numpy.ndarray`
    :arg r2: A (3,) array giving the coordinate of the second particle.
    :type r2: :class:`numpy.ndarray`
    :arg r10: A (3,) array giving the initial coordinate of the first particle.
    :type r10: :class:`numpy.ndarray`
    :arg r20: A (3,) array giving the initial coordinate of the second particle.
    :type r20: :class:`numpy.ndarray`

    :returns: The strain.
    :rtype: `numpy.float64`
    """
    return cstrain(r1, r2, r10, r20)


cdef inline float cstrain(float[:] r1, float[:] r2,
                           float[:] r10, float[:] r20):
    """
    C function for calculating the strain given current and initial
    coordinates.
    """
    cdef float l, dl

    l = ceuclid(r1, r2)
    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0


def strain2(l, r10, r20):
    """
    Calculate the strain between two particles given their current distance and
    initial positions.

    :arg l: The Euclidean distance between the two particles.
    :type l: `numpy.float64`
    :arg r10: A (3,) array giving the initial coordinate of the first particle.
    :type r10: :class:`numpy.ndarray`
    :arg r20: A (3,) array giving the initial coordinate of the second particle.
    :type r20: :class:`numpy.ndarray`

    :returns: The strain.
    :rtype: `numpy.float64`
    """
    return cstrain2(l, r10, r20)


cdef inline float cstrain2(float l, float[:] r10, float[:] r20):
    """
    C function for calculating the strain given the current displacement and
    the initial coordinates.
    """
    cdef float dl

    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0
