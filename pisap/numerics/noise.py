# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
#:Author: Samuel Farrens <samuel.farrens@gmail.com>
#:Version: 1.1
#:Date: 04/01/2017
##########################################################################

"""
This module contains methods for adding and removing noise from data.
"""

# System import
import numpy as np

# Package import
import pisap
from pisap.stats import sigma_mad
from pisap.numerics.linears import Dictionary


def add_noise(image, sigma=1.0, noise_type="gauss"):
    """ Add noise to data

    This method adds Gaussian or Poisson noise to the input data

    Parameters
    ----------
    data : np.ndarray or pisap.Image
        Input data array
    sigma : float, optional
        Standard deviation of the noise to be added ('gauss' only)
    noise_type : str {'gauss', 'poisson'}
        Type of noise to be added (default is 'gauss')

    Returns
    -------
    dirty_image: pisap.Image
        input data with added noise.

    Raises
    ------
    ValueError
        If `noise_type` is not 'gauss' or 'poisson'
    """
    if not isinstance(image, pisap.Image):
        image = pisap.Image(data=image)
    
    if noise_type not in ('gauss', 'poisson'):
        raise ValueError('Invalid noise type. Options are "gauss" or'
                         '"poisson"')

    if noise_type is 'gauss':
        image.data += sigma * np.random.randn(*image.data.shape)

    elif noise_type is 'poisson':
        image.data += np.random.poisson(np.abs(image.data))

    return image


def soft_thresholding(data, level):
    """ This method perfoms soft thresholding on the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    level : np.ndarray or float
        Threshold level

    Returns
    -------
    np.ndarray thresholded data
    """
    num = np.copy(data)
    num = np.maximum(np.abs(num) - level, 0)
    deno = num + level
    return (num / deno) * data


def hard_thresholding(data, level):
    """ This method perfoms hard thresholding on the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    level : np.ndarray or float
        Threshold level

    Returns
    -------
    np.ndarray thresholded data
    """
    return data * (np.abs(data) >= level)


def sigma_mad_sparse(grad_op, linear_op):
    """ Estimate the std from the mad routine on each approximation scale.

    Parameters
    ----------
    grad_op: instance
        Gradient operator.
    linear_op: instance
        Linear operator.

    Returns
    -------
    sigma: list
        a list of str estimate for each scale.
    """
    trf_grad = linear_op.op(grad_op.grad)
    return [sigma_mad(trf_grad.get_scale(ks)) for ks in range(trf_grad.nb_scale)]
