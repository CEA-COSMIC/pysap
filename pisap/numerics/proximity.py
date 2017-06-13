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
#:Date: 06/01/2017
##########################################################################

"""
This module contains classes of proximity operators for optimisation

"""

# System import
import numpy as np
import scipy.fftpack as pfft

# Package import
from .noise import soft_thresholding


class Positive(object):
    """Positivity proximity operator
    This class defines the positivity proximity operator
    """

    def __init__(self):
        pass

    def op(self, data, **kwargs):
        """ This method returns the location of positive coefficients in the
        input data 

        Parameters
        ----------
        data : np.ndarray
            Input data array
        **kwargs
            Arbitrary keyword arguments
        Returns
        -------
        np.ndarray all positive elements from input data
        """
        if np.issubsctype(data.dtype, np.complex):
            raise ValueError("can't compare complex value")
        return data * (data > 0)


class SoftThreshold(object):
    """ Soft threshold proximity operator

    This class defines the threshold proximity operator

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """
    def __init__(self, weights):
        self.update_weights(weights)

    def update_weights(self, weights):
        """ Update weights

        This method update the values of the weights

        Parameters
        ----------
        weights :DictionaryBase
            Input array of weights
        """
        self.weights = weights

    def op(self, data, extra_factor=1.0):
        """ Operator

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        DictionaryBase thresholded data

        """
        threshold = self.weights * extra_factor
        data._data = soft_thresholding(data._data, threshold._data)
        return data

