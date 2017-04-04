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
This module contains classses for defining algorithm operators and gradients.
Based on work by Yinghao Ge and Fred Ngole.
"""

# System import
import numpy as np
import scipy.fftpack as pfft


# Package import
from .algorithms import PowerMethod


class GradBase(object):
    """ Basic gradient class

    This class defines the basic methods that will be inherited by specific
    gradient classes
    """

    def MtMX(self, x):
        """ M^T M X

        This method calculates the action of the transpose of the matrix M on
        the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        Notes
        -----
        Calculates  M^T (MX)
        """
        return self.MtX(self.MX(x))

    def get_grad(self, x):
        """ Get the gradient step

        This method calculates the gradient step from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray gradient value

        Notes
        -----

        Calculates M^T (MX - Y)
        """
        self.grad = self.MtX(self.MX(x) - self.y)


class Grad2D(GradBase, PowerMethod):
    """ Standard 2D gradient class

    This class defines the operators for a 2D array

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    mask :  np.ndarray
        The subsampling mask.

    Notes
    -----
    The properties of `GradBase` and `PowerMethod` are inherited in this class
    """
    def __init__(self, data, mask):
        """ Initilize the Grad2D class.
        """
        # Set class attributes
        self.y = data
        self.mask = mask
        if mask is None:
            self.mask = np.ones(data.shape, dtype=int)

        # Inheritance
        PowerMethod.__init__(self, self.MtMX, self.y.shape)

    def MX(self, x):
        """ MX

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result
        """
        return self.mask * pfft.fft2(x)

    def MtX(self, x):
        """ MtX

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result
        """
        return np.real(pfft.ifft2(self.mask * x))


