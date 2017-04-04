##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
2D and nD Discrete Wavelet Transforms and Inverse Discrete Wavelet Transforms.
"""

# System import
from __future__ import division, print_function, absolute_import
from itertools import product
import numpy

# Package import
from .wtools import wavelets_per_axis
from .wtools import modes_per_axis
from .wtools import downcoef
from .wtools import upcoef
from .wtools import check_coeffs


def dwt1(data, wavelet, mode="symmetric", axis=-1):
    """ 1D Discrete Wavelet Transform.

    Parameters
    ----------
    data: array
        1-dimensional array with input data.
    wavelet: Wavelet, str, tuple of Wavelet or tuple of str
        the wavelet to be used.  This can also be a tuple containing a
        wavelet to apply along each axis.
    mode: str or tuple of str (optional, default 'symmetric')
        signal extension mode used in the decomposition,
        This can also be a tuple of modes specifying the mode to use on each
        axis.
    axis: int (optional, default -1)
        axes over which to compute the DWT.

    Returns
    -------
    (cA, cD): tuple
        approximation and detail coefficients.
    """
    # Check input parameters
    axes = (axis, )
    data = numpy.asarray(data)
    if data.ndim < len(numpy.unique(axes)):
        raise ValueError("Input array has fewer dimensions than the specified "
                         "axes")

    # DWT
    coeffs = dwtn(data, wavelet, mode, axes)

    return coeffs["a"], coeffs["d"]


def dwt2(data, wavelet, mode="symmetric", axes=(-2, -1)):
    """ 2D Discrete Wavelet Transform.

    Parameters
    ----------
    data: array
        2-dimensional array with input data.
    wavelet: Wavelet, str, tuple of Wavelet or tuple of str
        the wavelet to be used. This can also be a tuple containing a
        wavelet to apply along each axis.
    mode: str or tuple of str (optional, default 'symmetric')
        signal extension mode used in the decomposition,
        This can also be a tuple of modes specifying the mode to use on each
        axis.
    axes: tuple of int (optional, default (-2, -1))
        axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of 'None' (the
        default) selects all axes.

    Returns
    -------
    (cA, (cH, cV, cD)): tuple
        approximation, horizontal detail, vertical detail and diagonal
        detail coefficients respectively
        which sometimes is also interpreted as laid out in one 2D array
        of coefficients, where:

        .. code::

                                        -----------------
                                        |       |       |
                                        |cA(LL) |cH(HL) |
                                        |       |       |
            (cA, (cH, cV, cD))  <--->   -----------------
                                        |       |       |
                                        |cV(LH) |cD(HH) |
                                        |       |       |
                                        -----------------
    """
    # Check input parameters
    axes = tuple(axes)
    data = numpy.asarray(data)
    if len(axes) != 2:
        raise ValueError("Expect 2 axes.")
    if data.ndim < len(numpy.unique(axes)):
        raise ValueError("Input array has fewer dimensions than the specified "
                         "axes")

    # DWT
    coeffs = dwtn(data, wavelet, mode, axes)

    return coeffs["aa"], (coeffs["da"], coeffs["ad"], coeffs["dd"])


def dwtn(data, wavelet, mode="symmetric", axes=None):
    """ Single-level n-dimensional Discrete Wavelet Transform.

    Parameters
    ----------
    data: ndarray
        n-dimensional array with input data.
    wavelet: Wavelet, str, tuple of Wavelet or tuple of str
        the wavelet to be used.  This can also be a tuple containing a
        wavelet to apply along each axis.
    mode: str or tuple of str (optional, default 'symmetric')
        signal extension mode used in the decomposition,
        This can also be a tuple of modes specifying the mode to use on each
        axis. See numpy.pad for accepted values.
    axes: tuple of int (optional, deafault None)
        axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of 'None'
        selects all axes.

    Returns
    -------
    coeffs: dict
        results are arranged in a dictionary, where key specifies
        the transform type on each dimension and value is a n-dimensional
        coefficients array.

    For example, for a 2D case the result will look something like this::

        {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
         'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
         'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
         'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
        }

    For user-specified axes, the order of the characters in the
    dictionary keys map to the specified axes.
    """
    # Complex data special case
    data = numpy.asarray(data)
    if numpy.iscomplexobj(data):
        real = dwtn(data.real, wavelet, mode, axes)
        imag = dwtn(data.imag, wavelet, mode, axes)
        return dict((k, real[k] + 1j * imag[k]) for k in real.keys())

    # Check input data
    if data.dtype == numpy.dtype('object'):
        raise TypeError("Input must be a numeric array-like.")
    if data.ndim < 1:
        raise ValueError("Input data must be at least 1D.")

    # Define DWT axes
    if axes is None:
        axes = range(data.ndim)
    axes = [a + data.ndim if a < 0 else a for a in axes]

    # Define the modes and wavelets to be applied on each axis
    modes = modes_per_axis(mode, axes)
    wavelets = wavelets_per_axis(wavelet, axes)

    # nDWT
    coeffs = [("", data)]
    for axis, wav, mode in zip(axes, wavelets, modes):
        new_coeffs = []
        for subband, x in coeffs:
            cA = numpy.apply_along_axis(downcoef, axis, x, wav, mode, "a")
            cD = numpy.apply_along_axis(downcoef, axis, x, wav, mode, "d")
            new_coeffs.extend([(subband + "a", cA),
                               (subband + "d", cD)])
        coeffs = new_coeffs

    return dict(coeffs)


def idwt1(cA, cD, wavelet, mode="symmetric", axis=-1):
    """ 1-D Inverse Discrete Wavelet Transform.
    Reconstructs data from coefficient arrays.

    Parameters
    ----------
    cA: array or None
        approximation coefficients.  If None, will be considered as zero.
    cD : array or None
        detail coefficients.  If None, will be considered as zero.
    wavelet: Wavelet, str, tuple of Wavelet or tuple of str
        the wavelet to be used.  This can also be a tuple containing a
        wavelet to apply along each axis.
    mode: str or tuple of str (optional, default 'symmetric')
        signal extension mode used in the decomposition,
        This can also be a tuple of modes specifying the mode to use on each
        axis.
    axes: int (optional, deafault -1)
        axes over which to compute the DWT.

    Returns
    -------
    rec: array
       reconstruction of data from given coefficients.
    """
    # Check input parameters
    axes = (axis, )
    if cA is None and cD is None:
        raise ValueError("At least one coefficient parameter must be "
                         "specified.")
    coeffs = {"a": cA, "d": cD}

    # Drop the keys corresponding to value = None
    coeffs = dict((k, v) for k, v in coeffs.items() if v is not None)

    # iDWT
    rec = idwtn(coeffs, wavelet, mode, axes)

    return rec


def idwt2(coeffs, wavelet, mode="symmetric", axes=(-2, -1)):
    """ 2-D Inverse Discrete Wavelet Transform.
    Reconstructs data from coefficient arrays.

    Parameters
    ----------
    coeffs: tuple
        (cA, (cH, cV, cD)), a tuple with approximation coefficients and three
        details coefficients 2D arrays like from dwt2.

        The cefficents look something like this::


            .. code::

                {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
                 'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
                 'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
                 'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
                }

                                            -----------------
                                            |       |       |
                                            |cA(LL) |cH(HL) |
                                            |       |       |
                (cA, (cH, cV, cD))  <--->   -----------------
                                            |       |       |
                                            |cV(LH) |cD(HH) |
                                            |       |       |
                                            -----------------

    wavelet: Wavelet, str, tuple of Wavelet or tuple of str
        the wavelet to be used.  This can also be a tuple containing a
        wavelet to apply along each axis.
    mode: str or tuple of str (optional, default 'symmetric')
        signal extension mode used in the decomposition,
        This can also be a tuple of modes specifying the mode to use on each
        axis.
    axes: tuple of int (optional, deafault (-2, -1))
        axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of 'None' selects
        all axes.

    Returns
    -------
    rec: array
        the reconstructed data.
    """
    # Check input parameters
    cA, (cH, cV, cD) = coeffs
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes.")
    coeffs = {"aa": cA, "da": cH, "ad": cV, "dd": cD}

    # Drop the keys corresponding to value = None
    coeffs = dict((k, v) for k, v in coeffs.items() if v is not None)

    # iDWT
    rec = idwtn(coeffs, wavelet, mode, axes)

    return rec


def idwtn(coeffs, wavelet, mode="symmetric", axes=None):
    """ Single-level n-dimensional Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs: dict
        dictionary as in output of dwtn. Missing items will be treated
        as zeros.
    wavelet: Wavelet, str, tuple of Wavelet or tuple of str
        the wavelet to be used.  This can also be a tuple containing a
        wavelet to apply along each axis.
    mode: str or tuple of str (optional, default 'symmetric')
        signal extension mode used in the decomposition,
        This can also be a tuple of modes specifying the mode to use on each
        axis.
    axes: tuple of int (optional, deafault None)
        axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of 'None' selects
        all axes.

    Returns
    -------
    data: ndarray
        original signal reconstructed from input data.
    """
    # Check coefficients
    coeffs = check_coeffs(coeffs)

    # Complex data special case
    if any(numpy.iscomplexobj(v) for v in coeffs.values()):
        real_coeffs = dict((k, v.real) for k, v in coeffs.items())
        imag_coeffs = dict((k, v.imag) for k, v in coeffs.items())
        return (idwtn(real_coeffs, wavelet, mode, axes) -
                1j * idwtn(imag_coeffs, wavelet, mode, axes))

    # Check input parameters
    # > check key max length matches the number of axes transformed and the
    # band associated data
    ndim_transform = max(len(key) for key in coeffs.keys())
    try:
        coeff_shapes = (v.shape for k, v in coeffs.items()
                        if v is not None and len(k) == ndim_transform)
        coeff_shape = next(coeff_shapes)
    except StopIteration:
        raise ValueError("The 'coeffs' parameter must contain at least one "
                         "non-null wavelet band.")
    if any(shape != coeff_shape for shape in coeff_shapes):
        raise ValueError("The 'coeffs' parameter must all be of equal size "
                         "(or None)")

    # Define iDWT axes
    if axes is None:
        axes = range(ndim_transform)
        ndim = ndim_transform
    else:
        ndim = len(coeff_shape)
    axes = [a + ndim if a < 0 else a for a in axes]

    # Define the modes and wavelets to be applied on each axis
    modes = modes_per_axis(mode, axes)
    wavelets = wavelets_per_axis(wavelet, axes)

    # Define the number of values to take from the center of the idwtn for
    # each axis: calculate from mode to be the size of the original data,
    # rounded up to the nearest multiple of 2.
    takes = [2 * s for s in coeff_shape]

    # niDWT
    for band_length, (axis, wav, mode, take) in reversed(
            list(enumerate(zip(axes, wavelets, modes, takes)))):
        new_coeffs = {}
        new_keys = [
            "".join(subband) for subband in product("ad", repeat=band_length)]
        for subband in new_keys:
            cA = coeffs.get(subband + "a", None)
            cD = coeffs.get(subband + "d", None)
            if cA is not None:
                cA = numpy.apply_along_axis(
                    upcoef, axis, cA, wavelet=wav, take=take, part="a")
            if cD is not None:
                cD = numpy.apply_along_axis(
                    upcoef, axis, cD, wavelet=wav, take=take, part="d")
            if cA is None and cD is None:
                new_coeffs[subband] = None
            elif cD is None:
                new_coeffs[subband] = cA
            elif cA is None:
                new_coeffs[subband] = cD
            else:
                new_coeffs[subband] = cA + cD
        coeffs = new_coeffs

    return coeffs[""]
