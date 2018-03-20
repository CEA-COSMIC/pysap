##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
FISTA or CONDAT-VU MRI reconstruction.
"""


# System import
from __future__ import print_function
import copy
import time

# Package import
from .fourier import FFT2T
from .linear import Wavelet2T
# from pysap.plugins.mri.reconstruct.utils import unflatten
from pysap.plugins.mri.reconstruct.utils import fista_logo
from pysap.plugins.mri.reconstruct.cost import DualGapCost
from pysap.plugins.mri.reconstruct.reweight import mReweight
from pysap.plugins.mri.reconstruct.utils import condatvu_logo
from pysap.plugins.mri.reconstruct.gradient import GradAnalysis2
from pysap.plugins.mri.reconstruct.gradient import GradSynthesis2

# Third party import
import numpy as np
from modopt.math.stats import sigma_mad
from modopt.opt.linear import Identity
from modopt.opt.proximity import Positivity
from modopt.opt.proximity import SparseThreshold, LowRankMatrix
from modopt.opt.algorithms import Condat, ForwardBackward
from modopt.opt.reweight import cwbReweight


def sparse_rec_fista(data, wavelet_name, samples, mu, wavelet_name_t=None, nb_scales=4,
                     nb_scale_t=3, lambda_init=1.0, max_nb_of_iter=300, cost='l1',
                     atol=1e-4, non_cartesian=False, uniform_data_shape=None, verbose=0):
    """ The FISTA sparse reconstruction without reweightings.

    .. note:: At the moment, supports only 2D data.

    Parameters
    ----------
    data: ndarray
        the data to reconstruct (observation are expected in the Fourier
        space).
    wavelet_name: str
        the wavelet name to be used during the decomposition.
    samples: np.ndarray
        the mask samples in the Fourier domain.
    mu: float
       coefficient of regularization.
    nb_scales: int, default 4
        the number of scales in the wavelet decomposition.
    lambda_init: float, (default 1.0)
        initial value for the FISTA step.
    max_nb_of_iter: int (optional, default 300)
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    atol: float (optional, default 1e-4)
        tolerance threshold for convergence.
    non_cartesian: bool (optional, default False)
        if set, use the nfft rather than the fftw. Expect an 1D input dataset.
    uniform_data_shape: uplet (optional, default None)
        the shape of the matrix containing the uniform data. Only required
        for non-cartesian reconstructions.
    verbose: int (optional, default 0)
        the verbosity level.

    Returns
    -------
    x_final: ndarray
        the estimated FISTA solution.
    transform: a WaveletTransformBase derived instance
        the wavelet transformation instance.
    """
    # Check inputs
    start = time.clock()
    if non_cartesian and data.ndim != 1:
        raise ValueError("Expect 1D data with the non-cartesian option.")
    elif non_cartesian and uniform_data_shape is None:
        raise ValueError("Need to set the 'uniform_data_shape' parameter with "
                         "the non-cartesian option.")
    elif not non_cartesian and data.ndim != 2:
        raise ValueError("At the moment, this functuion only supports 2D "
                         "data.")

    # Define the gradient/linear/fourier operators
    linear_op = Wavelet2T(
        nb_scale=nb_scales,
        wavelet_name=wavelet_name,
        wavelet_name_t=wavelet_name_t,
        nb_scale_t=nb_scale_t)
    if non_cartesian:
        pass
        # fourier_op = NFFT2(
        #     samples=samples,
        #     shape=uniform_data_shape)
    else:
        fourier_op = FFT2T(
            samples=samples,
            shape=data.shape)
    gradient_op = GradSynthesis2(
        data=data,
        linear_op=linear_op,
        fourier_op=fourier_op)

    # Define the initial primal and dual solutions
    x_init = np.zeros(fourier_op.shape, dtype=np.complex)
    alpha = linear_op.op(x_init)
    alpha[...] = 0.0

    # Welcome message
    if verbose > 0:
        print(fista_logo())
        print(" - mu: ", mu)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", data.shape)
        print(" - wavelet: ", wavelet_name, "-", nb_scales)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - image variable shape: ", x_init.shape)
        print(" - alpha variable shape: ", alpha.shape)
        print("-" * 40)

    # Define the proximity dual operator
    weights = copy.deepcopy(alpha)
    weights[...] = mu
    if cost == 'l1':
        prox_op = SparseThreshold(linear_op, weights, thresh_type="soft")
    elif cost == 'low_rank':
        prox_op = LowRankMatrix(thresh=mu)
    else:
        raise AttributeError("cost has to be either 'l1' or 'low_rank'")

    # Define the optimizer
    cost_op = None
    opt = ForwardBackward(
        x=alpha,
        grad=gradient_op,
        prox=prox_op,
        cost=cost_op,
        auto_iterate=False)

    # Perform the reconstruction
    if verbose > 0:
        print("Starting optimization...")
    opt.iterate(max_iter=max_nb_of_iter)
    end = time.clock()
    if verbose > 0:
        # cost_op.plot_cost()
        # print(" - final iteration number: ", cost_op._iteration)
        # print(" - final log10 cost value: ", np.log10(cost_op.cost))
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    x_final = linear_op.adj_op(opt.x_final)

    return x_final, linear_op.transform



