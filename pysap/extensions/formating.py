# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains all the function to flatten properly the cube from a ISAP
.fits raw data.
"""

import numpy as np


###
# HELPERS

# GETTERS

def get_hbl(A):
    """ Return the half-bottom-left of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[l:, :l]


def get_hbr(A):
    """ Return the half-bottom-right of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[l:, l:]


def get_htl(A):
    """ Return the half-top-left of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[:l, :l]


def get_htr(A):
    """ Return the half-top-right of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[:l, l:]


def get_hr(A):
    """ Return the half-right of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[:, l:]


def get_hl(A):
    """ Return the half-left of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[:, :l]


def get_hb(A):
    """ Return the half-bottom of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[l:, :]


def get_ht(A):
    """ Return the half-top of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    return A[:l, :]

# SETTERS


def set_hbl(A, a):
    """ Return the half-bottom-left of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[l:, :l] = a


def set_hbr(A, a):
    """ Return the half-bottom-right of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[l:, l:] = a


def set_htl(A, a):
    """ Return the half-top-left of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[:l, :l] = a


def set_htr(A, a):
    """ Return the half-top-right of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[:l, l:] = a


def set_hr(A, a):
    """ Return the half-right of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[:, l:] = a


def set_hl(A, a):
    """ Return the half-left of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[:, :l] = a


def set_hb(A, a):
    """ Return the half-bottom of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[l:, :] = a


def set_ht(A, a):
    """ Return the half-top of the given array.
    """
    nx, _ = A.shape
    l = nx / 2
    A[:l, :] = a

###
# FLATTEN


def flatten_undecimated_n_bands(cube, trf):
    """ Flatten the decomposition coefficients from a 'cube' to a vector.
        'flatten_undecimated_n_bands' concern the 'cube' where each layer
        correspond to a
        undecimated band. We can have multiple bands per scale, which lead to
        nb_scale * nb_band_per_scale for one dimension.

        Parameters
        ----------
        cube: np.ndarray, the cube that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    return np.copy(cube.flatten())


def flatten_decimated_1_bands(cube, trf):
    """ Flatten the decomposition coefficients from a 'cube' to a vector.
        'flatten_decimated_1_bands' concern the 'cube' where it's actually a
        2d-array like
        the classic wavelet 2d-transform of 1 bands. It has the same formating
        than the 3 bands but the 'v' and 'h' bands or set to 0.

        Parameters
        ----------
        cube: np.ndarray, the cube that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    pieces = []
    for i in range(trf.nb_scale-1):
        pieces.append(get_htl(cube).flatten())
        cube = get_hbr(cube)
    pieces.append(cube.flatten())  # get approx
    return np.concatenate(pieces)


def flatten_decimated_3_bands(cube, trf):
    """ Flatten the decomposition coefficients from a 'cube' to a vector.
        'flatten_decimated_3_bands' concern the 'cube' where it's actually
        a 2d-array like
        the classic wavelet 2d-transform of 3 bands.

        Parameters
        ----------
        cube: np.ndarray, the cube that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    pieces = []
    for i in range(trf.nb_scale-1):
        pieces.append(get_htr(cube).flatten())
        pieces.append(get_hbr(cube).flatten())
        pieces.append(get_hbl(cube).flatten())
        cube = get_htl(cube)
    pieces.append(cube.flatten())  # get approx
    return np.concatenate(pieces)


def flatten_vector(cube, trf):
    """ Flatten the decomposition coefficients from a 'cube' to a vector.
        'flatten_vector' concern the 'curvelet-cube' where it's already
        a vector.

        Parameters
        ----------
        cube: np.ndarray, the cube that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    metadata_len = 1 + trf.nb_scale + 2 * trf.nb_band_per_scale.sum()
    data = np.zeros(len(cube) - metadata_len)
    cube_padd = 1 + trf.nb_scale + 2
    data_padd = 0
    for ks in range(trf.nb_scale):
        for kb in range(trf.nb_band_per_scale[ks]):
            tmp = cube[cube_padd:cube_padd+trf.bands_lengths[ks, kb]]
            data[data_padd:data_padd+trf.bands_lengths[ks, kb]] = tmp
            Nx = trf.bands_shapes[ks][kb][0]
            Ny = trf.bands_shapes[ks][kb][1]
            cube_padd += (Nx * Ny + 2)
            data_padd += (Nx * Ny)
    return data


def flatten_decimated_feauveau(cube, trf):
    """ Flatten decomposition coefficients from a 'cube' to a vector.
        'flatten_decimated_feauveau' concern the 'cube' where it's the Feauveau
        decimated...

        Parameters
        ----------
        cube: np.ndarray, the cube that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    pieces = []
    for i in range(trf.nb_scale-1):
        pieces.append(get_hbl(cube).flatten())
        pieces.append(get_hr(cube).flatten())
        cube = get_htl(cube)
    pieces.append(cube.flatten())  # get approx
    return np.concatenate(pieces)


###
# INFLATED


def inflated_undecimated_n_bands(trf):
    """ Inflated the decomposition coefficients from a vector to a 'cube'.
        'inflated_undecimated_n_bands' concern the vector where each layer
        correspond to a undecimated band. We can have multiple bands per scale,
        which lead to nb_scale * nb_band_per_scale for one dimension.

        Parameters
        ----------
        vector: np.ndarray, the vector that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    return np.copy(trf._analysis_data.reshape(trf._analysis_shape))


def inflated_decimated_1_bands(trf):
    """ Inflated the decomposition coefficients from a vector to a 'cube'.
        'inflated_decimated_1_bands' concern the vector where it's actually a
        2d-array like the classic wavelet 2d-transform of 1 bands. It has the
        same formating than the 3 bands but the 'v' and 'h' bands or set to 0.

        Parameters
        ----------
        vector: np.ndarray, the vector that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    cube = np.zeros(trf._analysis_shape, dtype=trf._analysis_data.dtype)
    tmp = cube
    for ks in range(trf.nb_scale-1):
        set_htl(tmp, trf[ks, 0])
        if ks == ((trf.nb_scale-1)-1):
            break
        else:
            tmp = get_hbr(tmp)
    set_hbr(tmp, trf[trf.nb_scale-1, 0])  # set approx
    return cube


def inflated_decimated_3_bands(trf):
    """ Inflated the decomposition coefficients from a vector to a 'cube'.
        'inflated_decimated_3_bands' concern the vector where it's actually a
        2d-array like the classic wavelet 2d-transform of 3 bands.

        Parameters
        ----------
        vector: np.ndarray, the vector that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    cube = np.zeros(trf._analysis_shape, dtype=trf._analysis_data.dtype)
    tmp = cube
    for ks in range(trf.nb_scale-1):
        set_htr(tmp, trf[ks, 0])
        set_hbr(tmp, trf[ks, 1])
        set_hbl(tmp, trf[ks, 2])
        if ks == ((trf.nb_scale-1)-1):
            break
        else:
            tmp = get_htl(tmp)
    set_htl(tmp, trf[trf.nb_scale-1, 0])  # set approx
    return cube


def inflated_vector(trf):
    """ Inflated the decomposition coefficients from a vector to a 'cube'.
        'inflated_vector' concern the vector where it's encode a curvelet
        Parameters
        ----------
        vector np.ndarray, the vector that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    metadata_len = 1 + trf.nb_scale + 2 * trf.nb_band_per_scale.sum()
    cube = np.zeros(len(trf._analysis_data) + metadata_len)
    cube[0] = trf.nb_scale
    cube[1:1+trf.nb_scale] = trf.nb_band_per_scale
    cube_padd = 1 + trf.nb_scale
    data_padd = 0
    for ks in range(trf.nb_scale):
        for kb in range(trf.nb_band_per_scale[ks]):
            Nx = trf.bands_shapes[ks][kb][0]
            Ny = trf.bands_shapes[ks][kb][1]
            cube[cube_padd] = Nx
            cube_padd += 1
            cube[cube_padd] = Ny
            cube_padd += 1
            tmp = trf[ks, kb].flatten()
            cube[cube_padd:cube_padd+trf.bands_lengths[ks, kb]] = tmp
            cube_padd += (Nx * Ny)
            data_padd += (Nx * Ny)
    return cube


def inflated_decimated_feauveau(trf):
    """ Inflated the decomposition coefficients from a vector to a 'cube'.
        'inflated_decimated_feauveau' concern the vector where it's the
        Feauveau decimated...

        Parameters
        ----------
        vector: np.ndarray, the vector that containes the decomposition
        coefficients.

        Return:
        --------
        data: np.ndarray, the flatten 'cube'.
    """
    cube = np.zeros(trf._analysis_shape, dtype=trf._analysis_data.dtype)
    tmp = cube
    for ks in range(trf.nb_scale-1):
        set_hbl(tmp, trf[ks, 0])
        set_hr(tmp, trf[ks, 1])
        if ks == ((trf.nb_scale-1)-1):
            break
        else:
            tmp = get_htl(tmp)
    set_htl(tmp, trf[trf.nb_scale-1, 0])  # set approx
    return cube

###
# FORMATING FCTS INDEXES


FLATTENING_FCTS = [flatten_undecimated_n_bands,
                   flatten_decimated_1_bands,
                   flatten_decimated_3_bands,
                   flatten_vector,
                   flatten_decimated_feauveau]

INFLATING_FCTS = [inflated_undecimated_n_bands,
                  inflated_decimated_1_bands,
                  inflated_decimated_3_bands,
                  inflated_vector,
                  inflated_decimated_feauveau]
