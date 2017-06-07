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


def denoise(data, level, threshold_type='hard'):
    """ Remove noise from data

    This method perfoms hard or soft thresholding on the input data

    Parameters
    ----------
    data : np.ndarray
        Input data array
    level : float
        Threshold level
    threshold_type : str {'hard', 'soft'}
        Type of noise to be added (default is 'hard')

    Returns
    -------
    np.ndarray thresholded data

    Raises
    ------
    ValueError
        If `threshold_type` is not 'hard' or 'soft'
    """
    if threshold_type not in ('hard', 'soft'):
        raise ValueError('Invalid threshold type. Options are "hard" or'
                         '"soft"')

    if threshold_type == 'soft':
        if data.is_complex:
            #sz = max( abs(z) - T , 0 ) / ( max( abs(z) - T, 0) + T ) * z
            deno = (((data.absolute - level) >= 0)._data.max() + level) * data
            num = ((data.absolute - level) >= 0)._data.max()
            return deno / num
        else:
            return data.sign * (data.absolute - level) * (data.absolute >= level)
    else:
        return data * (data.absolute >= level)


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
    return [sigma_mad(scale_data) for scale_data in linear_op.op(grad_op.grad)]
