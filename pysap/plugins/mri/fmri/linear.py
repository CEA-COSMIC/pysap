# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Defines linear operators for fMRI
"""

# Package imports
from builtins import zip
from .transform import TransformT

# Third party import
import numpy as np


class Wavelet2T(object):
    """
    Wavelet transform class for 2D+T data
    """
    def __init__(self, wavelet_name, nb_scale=4, wavelet_name_t=None,
                 nb_scale_t=1, verbose=0):
        """
        Class initialization
        :param wavelet_name: str
            2D Wavelet name
        :param nb_scale: int, default 4
            Number of decomposition scales for the 2D transform
        :param wavelet_name_t: str, default None
            1D Wavelet name
        :param nb_scale_t: int, default 1
            Number of decomposition scales for the 1D transform
        :param verbose: int, default 0
            verbosity level
        """
        self.nb_scale = nb_scale
        self.nb_scale_t = nb_scale_t
        self.transform = TransformT(wavelet_name, nb_scale,
                                    wavelet_name_t, nb_scale_t, verbose)
        self.coeffs_shape = None

    def op(self, data):
        """
        Defines the wavelet operator
        :param data: np.ndarray
            data to be transformed
        :return: coeffs: np.ndarray
            the coefficients of the transformed data
        """
        coeffs, self.coeffs_shape = self.transform.analysis(data)
        return coeffs

    def adj_op(self, coeffs):
        """
        Defines the wavelet adjoint operator
        :param coeffs: np.ndarray
            Wavelet coefficients
        :return: image
            The reconstructed image
        """
        image = self.transform.synthesis(coeffs)
        return image

    def l2norm(self, shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        shape: uplet
            the data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        fake_data[list(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)
