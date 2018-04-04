# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
CONDA-VU Galaxy Image Deconvolution
"""

# System import
from __future__ import print_function
from builtins import range, zip
import pysap
from pysap.numerics.linear import WaveletConvolve2
from pysap.plugins.astro.deconvolve.wavelet_filters import get_cospy_filters
from pysap.utils import condatvu_logo

# Third party import
import numpy as np
from modopt.base.np_adjust import rotate
from modopt.opt.algorithms import Condat
from modopt.opt.cost import costObj
from modopt.opt.gradient import GradBasic
from modopt.opt.proximity import Positivity, SparseThreshold
from modopt.opt.reweight import cwbReweight
from modopt.math.convolve import convolve
from modopt.math.stats import sigma_mad
from modopt.signal.wavelet import filter_convolve


def psf_convolve(data, psf, psf_rot=False):
    """PSF Convolution

    Parameters
    ----------
    data : np.ndarray
        Input data, 2D image
    psf : np.ndarray
        Input PSF, 2D image
    psf_rot : bool, optional
        Option to rotate the input PSF (default is False)

    Returns
    -------
    np.ndarray convolved image

    """

    if psf_rot:
        psf = rotate(psf)

    return convolve(data, psf)


def get_weights(data, psf, filters, wave_thresh_factor=np.array([3, 3, 4])):
    """Get Sparsity Weights

    Parameters
    ----------
    data : np.ndarray
        Input data, 2D image
    psf : np.ndarray
        Input PSF, 2D image
    filters : np.ndarray
        Wavelet filters
    wave_thresh_factor : np.ndarray, optional
        Threshold factors for each wavelet scale (default is
        np.array([3, 3, 4]))

    Returns
    -------
    np.ndarray weights

    """

    noise_est = sigma_mad(data)

    filter_conv = filter_convolve(np.rot90(psf, 2), filters)

    filter_norm = np.array([np.linalg.norm(a) * b * np.ones(data.shape)
                            for a, b in zip(filter_conv, wave_thresh_factor)])

    return noise_est * filter_norm


def sparse_deconv_condatvu(data, psf, n_iter=300, n_reweights=1):
    """Sparse Deconvolution with Condat-Vu

    Parameters
    ----------
    data : np.ndarray
        Input data, 2D image
    psf : np.ndarray
        Input PSF, 2D image
    n_iter : int, optional
        Maximum number of iterations
    n_reweights : int, optional
        Number of reweightings

    Returns
    -------
    np.ndarray deconvolved image

    """

    # Print the algorithm set-up
    print(condatvu_logo())

    # Define the wavelet filters
    filters = (get_cospy_filters(data.shape,
               transform_name='LinearWaveletTransformATrousAlgorithm'))

    # Set the reweighting scheme
    reweight = cwbReweight(get_weights(data, psf, filters))

    # Set the initial variable values
    primal = np.ones(data.shape)
    dual = np.ones(filters.shape)

    # Set the gradient operators
    grad_op = GradBasic(data, lambda x: psf_convolve(x, psf),
                        lambda x: psf_convolve(x, psf, psf_rot=True))

    # Set the linear operator
    linear_op = WaveletConvolve2(filters)

    # Set the proximity operators
    prox_op = Positivity()
    prox_dual_op = SparseThreshold(linear_op, reweight.weights)

    # Set the cost function
    cost_op = costObj([grad_op, prox_op, prox_dual_op], tolerance=1e-6,
                      cost_interval=1, plot_output=True, verbose=False)

    # Set the optimisation algorithm
    alg = Condat(primal, dual, grad_op, prox_op, prox_dual_op, linear_op,
                 cost_op, rho=0.8, sigma=0.5, tau=0.5, auto_iterate=False)

    # Run the algorithm
    alg.iterate(max_iter=n_iter)

    # Implement reweigting
    for rw_num in range(n_reweights):
        print(' - Reweighting: {}'.format(rw_num + 1))
        reweight.reweight(linear_op.op(alg.x_final))
        alg.iterate(max_iter=n_iter)

    # Return the final result
    return alg.x_final
