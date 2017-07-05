# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classses for defining algorithm operators and gradients.
Based on work by Yinghao Ge and Fred Ngole.
"""

# System import
import numpy as np
import scipy.fftpack as pfft

# Third party import
from sf_deconvolve.lib.gradient import GradBasic
from sf_deconvolve.lib.algorithms import PowerMethod


class Grad2DSynthese(GradBasic, PowerMethod):
    """ Standard 2D gradient.

    This class defines the operators for a 2D array.
    """
    def __init__(self, data, mask):
        """ Initilize the Grad2DSynthese class.

        Parameters
        ----------
        data: np.ndarray
            Input data array, an array of 2D observed images (i.e. with noise)
        mask:  np.ndarray
            The subsampling mask.
        """
        self.y = data
        self.mask = mask
        if mask is None:
            self.mask = np.ones(data.shape, dtype=int)
        PowerMethod.__init__(self, self.MtMX, self.y.shape)
        self.get_spec_rad()

    def set_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values
        """
        return np.random.random(self.y.shape).astype(np.complex)

    def MX(self, x):
        """ MX.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: np.ndarray
            Input data array.

        Returns
        -------
        coeffs: np.ndarray
            Fourier transform coefficents.
        """
        return self.mask * pfft.fft2(x)

    def MtX(self, coeffs):
        """ MtX.

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data

        Parameters
        ----------
        coeffs: np.ndarray
            Fourier transform coefficents.

        Returns
        -------
        x: np.ndarray
            Reconstructed data array
        """
        return pfft.ifft2(self.mask * coeffs)


class Grad2DAnalyse(GradBasic, PowerMethod):
    """ Analysis 2D gradient class.

    This class defines the grad operators for |M*F*invL*alpha - data|**2.
    """
    def __init__(self, data, mask, linear_operator):
        """ Initilize the Grad2DAnalyse class.

        Parameters
        ----------
        data: np.ndarray
            Input data array, an array of 2D observed images (i.e. with noise)
        mask:  np.ndarray
            The subsampling mask.
        linear_operator: pisap.numeric.linear.Wavelet
            A linear operator.
        """
        self.y = data
        self.mask = mask
        self.linear_operator = linear_operator
        if mask is None:
            self.mask = np.ones(data.shape, dtype=int)
        PowerMethod.__init__(self, self.MtMX, self.y.shape)
        self.get_spec_rad()

    def set_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values
        """
        fake_data = np.zeros(self.y.shape).astype(np.complex)
        coeffs = self.linear_operator.op(fake_data)
        coeffs = np.random.random(len(coeffs)).astype(np.complex)
        return coeffs

    def MX(self, x):
        """ MX.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: nd-array
            Input decomposisiton coefficients.

        Returns
        -------
        coeffs: np.ndarray
            Fourier transform coefficents.
        """
        return self.mask * pfft.fft2(self.linear_operator.adj_op(x))

    def MtX(self, coeffs):
        """ MtX.

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data in
        the frequency domain.

        Parameters
        ----------
        x: np.ndarray
            Input data array, an array of recovered 2D kspace

        Returns
        -------
        x: nd-array
            Reconstructed data array decomposisiton coefficients.
        """
        return self.linear_operator.op(pfft.ifft2(self.mask * coeffs))
