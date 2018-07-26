# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
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


def convert_mask_to_locations(mask):
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
    row, col = np.where(mask == 1)
    row = row.astype("float") / mask.shape[0] - 0.5
    col = col.astype("float") / mask.shape[1] - 0.5
    return np.c_[row, col]


def convert_locations_to_mask(samples_locations, img_shape):
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
    samples_locations = samples_locations.astype("float")
    samples_locations += 0.5
    samples_locations[:, 0] *= img_shape[0]
    samples_locations[:, 1] *= img_shape[1]
    samples_locations = np.floor(samples_locations)
    samples_locations = samples_locations.astype("int")
    mask = np.zeros(img_shape)
    mask[samples_locations[:, 0], samples_locations[:, 1]] = 1
    return mask


def normalize_samples(samples):
    """
    This function normalize the samples so it can be between [-0.5; 0.5[ for
    the non-cartesian case

    Parameters:
    -----------
    samples: np.ndarray
        Unnormalized samples

    Return:
    -------
    normalized_samples: np.ndarray
        Same shape as the parameters but with values between [-0.5; 0.5[
    """
    samples_locations = np.copy(samples.astype('float'))
    samples_locations[:, 0] /= 2 * np.abs(samples_locations[:, 0]).max()
    samples_locations[:, 1] /= 2 * np.abs(samples_locations[:, 1]).max()
    if samples_locations.max() == 0.5:
        dim1, dim2 = np.where(samples_locations == 0.5)
        samples_locations[dim1, dim2] = -0.5
    return samples_locations


def generate_operators(data, wavelet_name, samples, nb_scales=4,
                       non_cartesian=False, uniform_data_shape=None,
                       gradient_space="analysis"):
    """ Function that ease the creation of a set of common operators.

    .. note:: At the moment, supports only 2D data.

    Parameters
    ----------
    data: ndarray
        the data to reconstruct: observation are expected in Fourier space.
    wavelet_name: str
        the wavelet name to be used during the decomposition.
    samples: np.ndarray
        the mask samples in the Fourier domain.
    nb_scales: int, default 4
        the number of scales in the wavelet decomposition.
    non_cartesian: bool (optional, default False)
        if set, use the nfftw rather than the fftw. Expect an 1D input dataset.
    uniform_data_shape: uplet (optional, default None)
        the shape of the matrix containing the uniform data. Only required
        for non-cartesian reconstructions.
    gradient_space: str (optional, default 'analysis')
        the space where the gradient operator is defined: 'analysis' or
        'synthesis'

    Returns
    -------
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    prox_op: instance of ProximityParent
        the proximal operator.
    cost_op: instance of costObj
        the cost function used to check for convergence during the
        optimization.
    """
    # Local imports
    from pysap.numerics.cost import DualGapCost
    from pysap.numerics.linear import Wavelet2
    from pysap.numerics.fourier import FFT2
    from pysap.numerics.fourier import NFFT2
    from pysap.numerics.gradient import GradAnalysis2
    from pysap.numerics.gradient import GradSynthesis2
    from modopt.opt.proximity import SparseThreshold

    # Check input parameters
    if gradient_space not in ("analysis", "synthesis"):
        raise ValueError("Unsupported gradient space '{0}'.".format(
            gradient_space))
    if non_cartesian and data.ndim != 1:
        raise ValueError("Expect 1D data with the non-cartesian option.")
    elif non_cartesian and uniform_data_shape is None:
        raise ValueError("Need to set the 'uniform_data_shape' parameter with "
                         "the non-cartesian option.")
    elif not non_cartesian and data.ndim != 2:
        raise ValueError("At the moment, this functuion only supports 2D "
                         "data.")

    # Define the gradient/linear/fourier operators
    linear_op = Wavelet2(
        nb_scale=nb_scales,
        wavelet_name=wavelet_name)
    if non_cartesian:
        fourier_op = NFFT2(
            samples=samples,
            shape=uniform_data_shape)
    else:
        fourier_op = FFT2(
            samples=samples,
            shape=data.shape)
    if gradient_space == "synthesis":
        gradient_op = GradSynthesis2(
            data=data,
            linear_op=linear_op,
            fourier_op=fourier_op)
    else:
        gradient_op = GradAnalysis2(
            data=data,
            fourier_op=fourier_op)

    # Define the proximity dual/primal operator
    prox_op = SparseThreshold(linear_op, None, thresh_type="soft")

    # Define the cost function
    if gradient_space == "synthesis":
        cost_op = None
    else:
        cost_op = DualGapCost(
            linear_op=linear_op,
            initial_cost=1e6,
            tolerance=1e-4,
            cost_interval=1,
            test_range=4,
            verbose=0,
            plot_output=None)

    return gradient_op, linear_op, prox_op, cost_op
