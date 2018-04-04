# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Reweighting optimisation strategies.
"""


# Package import
from pysap.base.utils import flatten

# Third party import
import numpy as np
from modopt.math.stats import sigma_mad


class mReweight(object):
    """ Ming reweighting.

    This class implements the reweighting scheme described in Ming2017.

    Parameters
    ----------
    weights: ndarray
        Array of weights
    linear_op: pysap.numeric.linear.Wavelet
        A linear operator.
    thresh_factor: float, default 1
        Threshold factor: sigma threshold.
    """
    def __init__(self, weights, linear_op, thresh_factor=1):
        self.weights = weights
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor
        self.linear_op = linear_op

    def reweight(self, x_new):
        """ Updat the weights.

        Parameters
        ----------
        x_new: ndarray
            the current primal solution.

        Returns
        -------
        sigma_est: ndarray
            the variance estimate on each scale.
        """
        self.linear_op.op(x_new)
        weights = np.empty((0, ), dtype=self.weights.dtype)
        sigma_est = []
        for scale in range(self.linear_op.transform.nb_scale):
            bands_array, _ = flatten(self.linear_op.transform[scale])
            if scale == (self.linear_op.transform.nb_scale - 1):
                std_at_scale_i = 0.
            else:
                std_at_scale_i = sigma_mad(bands_array)
            sigma_est.append(std_at_scale_i)
            thr = np.ones(bands_array.shape, dtype=weights.dtype)
            thr *= self.thresh_factor * std_at_scale_i
            weights = np.concatenate((weights, thr))
        self.weights = weights
        return sigma_est
