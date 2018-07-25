##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common tools for MRI image reconstruction.
"""


# System import
import numpy as np
import matplotlib.pyplot as plt


def convert_mask_to_locations_3D(mask):
    """ Return the converted Cartesian mask as sampling locations.

    Parameters
    ----------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.

    Returns
    -------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    """
    dim1, dim2, dim3 = np.where(mask == 1)
    dim1 = dim1.astype("float") / mask.shape[0] - 0.5
    dim2 = dim2.astype("float") / mask.shape[1] - 0.5
    dim3 = dim3.astype("float") / mask.shape[2] - 0.5
    return np.c_[dim1, dim2, dim3]


def normalize_samples(samples_locations):
    """Normalize the 3D samples between [-.5; .5[

    Parameters
    ----------
    samples_locations: np.array
        A representation of the 3D locations of the samples
    """
    samples_locations = samples_locations.astype('float')
    samples_locations[:, 0] /= 2 * np.abs(samples_locations[:, 0]).max()
    samples_locations[:, 1] /= 2 * np.abs(samples_locations[:, 1]).max()
    samples_locations[:, 2] /= 2 * np.abs(samples_locations[:, 2]).max()
    while samples_locations.max() == 0.5:
        dim1, dim2 = np.where(samples_locations == 0.5)
        samples_locations[dim1, dim2] = -0.5
    return samples_locations


def convert_locations_to_mask_3D(samples_locations, img_shape):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters
    ----------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    img_shape: tuple of int
        shape of the desired mask, not necessarly a square matrix.

    Returns
    -------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.
    """
    samples_locations = np.copy(samples_locations).astype("float")
    samples_locations += 0.5
    samples_locations[:, 0] *= img_shape[0]
    samples_locations[:, 1] *= img_shape[1]
    samples_locations[:, 2] *= img_shape[2]
    samples_locations = np.round(samples_locations) - 1
    samples_locations = samples_locations.astype("int")
    mask = np.zeros(img_shape)
    mask[samples_locations[:, 0],
         samples_locations[:, 1],
         samples_locations[:, 2]] = 1
    return mask
