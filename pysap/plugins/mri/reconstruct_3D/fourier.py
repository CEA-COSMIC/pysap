# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 3017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""


# Package import
from .utils import convert_locations_to_mask
from pysap.plugins.mri.reconstruct.fourier import FourierBase

# Third party import
import pynfft
import numpy as np
import scipy.fftpack as pfft


class FFT3(FourierBase):
    """ Standard 3D Fast Fourrier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """
    def __init__(self, samples, shape):
        """ Initilize the 'FFT3' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        """
        self.samples = samples
        self.shape = shape
        self._mask = convert_locations_to_mask(self.samples, self.shape)

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        return self._mask * pfft.fftn(img)

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 3-D
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        return pfft.ifftn(self._mask * x)


class NFFT3(FourierBase):
    """ Standard 3D non catesian Fast Fourrier Transform class

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """

    def __init__(self, samples, shape):
        """ Initilize the 'NFFT3' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        """
        self.plan = pynfft.NFFT(N=shape, M=len(samples))
        self.shape = shape
        self.samples = samples
        self.plan.x = self.samples
        self.plan.precompute()

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        self.plan.f_hat = img
        return (1.0 / np.sqrt(self.plan.M)) * self.plan.trafo()

    def adj_op(self, x):
        """ This method calculates inverse masked non-cartesian Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-cartesian Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        self.plan.f = x
        return (1.0 / np.sqrt(self.plan.M)) * self.plan.adjoint()
