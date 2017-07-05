##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classes for reweighting optimisation implementations.
"""

# System import
import numpy as np


class mReweight(object):
    """ Ming reweighting.

    This class implements the reweighting scheme described in Ming2017.

    Parameters
    ----------
    weights: ndarray
        Array of weights
    linear_operator: pisap.numeric.linear.Wavelet
        A linear operator.
    thresh_factor: float, default 1
        Threshold factor: sigma threshold.
    """
    def __init__(self, weights, linear_operator, thresh_factor=1):
        self.weights = weights
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor
        self.linear_operator = linear_operator

    def reweight(self, sigma_est, data):
        """ Reweight.

        Parameters
        ----------
        sigma_est: ndarray
            the variance estimate on each scale.
        data: ndarray
            the wavelet coefficients.
        """
        weights = np.ones_like(data)
        for scale in range(self.linear_operator.transform.nb_scale):
            thr = self.thresh_factor * sigma_est[scale]
            scale_indices = (
                self.linear_operator.transform.scales_padds[scale],
                self.linear_operator.transform.scales_padds[scale + 1])
            scale_data = data[scale_indices[0]: scale_indices[1]]
            indices = scale_data > thr
            weights[scale_indices[0]: scale_indices[1]][indices] = (
                thr / np.abs(scale_data[indices]))
        self.weights = weights
