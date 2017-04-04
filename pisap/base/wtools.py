##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import sys
import numpy
from collections import Iterable

# Package import
from .wavelet import Wavelet

# Python 2/3 compatibility
if sys.version_info[0] == 3:
    types = (str, )
else:
    types = (basestring, )


def _as_wavelet(wavelet):
    """ Convert wavelet name to a Wavelet object.

    Parameters
    ----------
    wavelet: str or Wavelet
        the wavelet or the wavelet name.

    Returns
    -------
    wavelet: Wavelet
        the wavelet.
    """
    if not isinstance(wavelet, Wavelet):
        wavelet = Wavelet(wavelet)
    return wavelet


def wavelets_per_axis(wavelet, axes):
    """ Initialize Wavelets for each axis to be transformed.

    Parameters
    ----------
    wavelet: str, Wavelet, tuple of str or tuple of Wavelet
        if a single Wavelet is provided, it will used for all axes. Otherwise
        one Wavelet per axis must be provided.
    axes: list
        the tuple of axes to be transformed.

    Returns
    -------
    wavelets: list of Wavelet objects
        a tuple of Wavelets equal in length to 'axes'.

    """
    axes = tuple(axes)
    # Same wavelet on all axes
    if isinstance(wavelet, types + (Wavelet, )):
        wavelets = [_as_wavelet(wavelet), ] * len(axes)

    # Potentially different wavelet per axis
    elif isinstance(wavelet, Iterable):
        if len(wavelet) == 1:
            wavelets = [_as_wavelet(wavelet[0]), ] * len(axes)
        else:
            if len(wavelet) != len(axes):
                raise ValueError((
                    "The number of wavelets must match the number of axes "
                    "to be transformed."))
            wavelets = [_as_wavelet(w) for w in wavelet]
    else:
        raise ValueError("The 'wavelet' parameter must be a string, Wavelet "
                         "or iterable")
    return wavelets


def modes_per_axis(modes, axes):
    """ Initialize mode for each axis to be transformed.

    Parameters
    ----------
    modes: str or tuple of str
        if a single mode is provided, it will used for all axes. Otherwise
        one mode per axis must be provided.
    axes: tuple
        the tuple of axes to be transformed.

    Returns
    -------
    modes : tuple of int
        a tuple of modes equal in length to 'axes'.

    """
    axes = tuple(axes)
    # Same mode on all axes
    if isinstance(modes, types):
        modes = [modes, ] * len(axes)
    # Potentially different mode per axis
    elif isinstance(modes, Iterable):
        if len(modes) == 1:
            modes = [modes, ] * len(axes)
        else:
            if len(modes) != len(axes):
                raise ValueError(("The number of modes must match the number "
                                  "of axes to be transformed."))
        modes = modes
    else:
        raise ValueError("The 'modes' parameter must be a string or iterable")
    return modes


def check_coeffs(coeffs):
    """ Check that the DWT coefficients are valid.

    Parameters
    ----------
    coeffs: dict
        dictionary as in output of dwtn.

    Returns
    -------
    checked_coeffs: dict
        the checkes dwt coefficients.
    """
    # Check None items
    missing_keys = [k for k, v in coeffs.items() if v is None]
    if missing_keys:
        raise ValueError(
            "The following detail coefficients were set to 'None': "
            "{0}.".format(missing_keys))

    # Check key names
    invalid_keys = [k for k, v in coeffs.items() if not set(k) <= set("ad")]
    if invalid_keys:
        raise ValueError(
            "The following invalid keys were found in the detail "
            "coefficient dictionary: {0}.".format(invalid_keys))

    # Check key length
    key_lengths = [len(k) for k in coeffs.keys()]
    if len(numpy.unique(key_lengths)) > 1:
        raise ValueError(
            "All detail coefficient names must have equal length.")

    return dict((k, numpy.asarray(v)) for k, v in coeffs.items())


def downcoef(data, wavelet, mode="symmetric", part="a", scale=1):
    """ Partial Discrete Wavelet Transform data decomposition.
    Useful when you need only approximation or only details at the given level.

    Parameters
    ----------
    data: array
        input signal.
    wavelet: Wavelet or str
        wavelet to use.
    mode: str (optional, default "symmetric")
        Signal extension mode, see `Modes`.  Default is 'symmetric'.
    part: str (optional, default 'a')
        Coefficients type:
        * 'a' - approximations reconstruction is performed
        * 'd' - details reconstruction is performed
    scale: int (optional, default 1)
        the decomposition scale. 

    Returns
    -------
    coeffs : ndarray
        1-D array of coefficients.
    """
    # Complex data case
    if numpy.iscomplexobj(data):
        return (downcoef(data.real, wavelet, mode, part, level) +
                1j * downcoef(data.imag, wavelet, mode, part, level))

    # Check input parameters
    if part not in "ad":
        raise ValueError(
            "Argument 1 must be 'a' or 'd', not {0}.".format(part))
    if scale < 1:
        raise ValueError("Value of scale must be greater than 0.")
    wavelet = _as_wavelet(wavelet)

    # Perform the partial discrete wavelet transform decomposition
    for i in range(scale):
        if (i < scale - 1):
            coeffs = downsampling_convolution(data, wavelet, part="a", mode=mode)
        else:
            coeffs = downsampling_convolution(data, wavelet, part, mode)
        data = coeffs

    return coeffs


def downsampling_convolution(data, wavelet, part, mode="symmetric"):
    """ 1D convolutation and downsampling.

    Parameters
    ----------
    data: array
        input signal.
    wavelet: Wavelet or str
        wavelet to use.
    part: str
        Coefficients type:
        * 'a' - approximations reconstruction is performed
        * 'd' - details reconstruction is performed
    mode: str (optional, default "symmetric")
        Signal extension mode, see `Modes`.  Default is 'symmetric'.

    Returns
    -------
    coeffs: array
        the downsampled convolution coefficients: cA or cD deppending on input
        requested part.
    """
    # Switch on part
    size = numpy.size(data)
    if part == "a":
        len_filter = wavelet.dec_len
        kernel = wavelet.dec_lo
    else:
        len_filter = wavelet.dec_len
        kernel = wavelet.dec_hi
        
    # Pads the data
    padded_data = numpy.pad(data, (len_filter - 1, len_filter - 1), mode)

    # Convolution
    coeffs = numpy.convolve(padded_data, kernel, "valid")

    # Downsample by 2
    coeffs = coeffs[len_filter - 1: size + len_filter + 1: 2]

    return coeffs


def upcoef(coeffs, wavelet, part="a", scale=1, take=0):
    """  Direct reconstruction from coefficients.

    Parameters
    ----------
    coeffs: array
        coefficients array to recontruct.
    wavelet: Wavelet or str
        wavelet to use.
    part: str (optional, default 'a')
        Coefficients type:
        * 'a' - approximations reconstruction is performed
        * 'd' - details reconstruction is performed
    scale: int (optional, default 1)
        the decomposition scale. 
    take: int (optional, default 0)
        take central part of length equal to 'take' from the result.

    Returns
    -------
    rec: ndarray
        1-D array with reconstructed data from coefficients.
    """
    # Complex data case
    if numpy.iscomplexobj(coeffs):
        return (upcoef(coeffs.real, wavelet, mode, part, level, take) +
                1j * upcoef(coeffs.imag, wavelet, mode, part, level, take))

    # Check input parameters
    if part not in "ad":
        raise ValueError(
            "Argument 1 must be 'a' or 'd', not {0}.".format(part))
    if scale < 1:
        raise ValueError("Value of scale must be greater than 0.")
    wavelet = _as_wavelet(wavelet)

    # Perform the partial discrete wavelet transform reconstruction
    for i in range(scale):
        if i > 0:
            rec = upsampling_convolution(coeffs, wavelet, part="a", take=take)
        else:
            rec = upsampling_convolution(coeffs, wavelet, part, take)
        coeffs = rec

    return rec


def upsampling_convolution(coeffs, wavelet, part="a", take=0):
    """ 1D convolutation and upsampling.

    Parameters
    ----------
    coeffs: array
        coefficients array to recontruct.
    wavelet: Wavelet or str
        wavelet to use.
    part: str (optional, default 'a')
        Coefficients type:
        * 'a' - approximations reconstruction is performed
        * 'd' - details reconstruction is performed
    take: int (optional, default 0)
        take central part of length equal to 'take' from the result.

    Returns
    -------
    rec: ndarray
        1-D array with reconstructed data from coefficients.
    """
    # Switch on part
    size = numpy.size(coeffs)
    if part == "a":
        len_filter = wavelet.rec_len
        kernel = wavelet.rec_lo
    else:
        len_filter = wavelet.rec_len
        kernel = wavelet.rec_hi
    rec_len = 2 * size + len_filter - 1

    # Up sample by 2
    cup = numpy.zeros(2 * size)
    cup[::2] = coeffs

    # Convolution
    rec = numpy.convolve(cup, kernel, "full")

    # Restore the size of the original data
    if take > 0 and take < rec_len:
        left_bound = right_bound = (rec_len - take) // 2
        if (rec_len - take) % 2:
            # right_bound must never be zero for indexing to work
            right_bound = right_bound + 1

        return rec[left_bound: -right_bound]

    return rec
