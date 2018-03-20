# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
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

# Third party import
try:
    import pynfft
except Exception:
    pass
import numpy as np
import scipy.fftpack as pfft
from pysap.plugins.mri.reconstruct.fourier import FourierBase


class FFT2T(FourierBase):
    """ Standard 2D+T Fast Fourier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarily a square matrix).
    """
    def __init__(self, samples, shape):
        """ Initialize the 'FFT2' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (not necessarily a square matrix).
        """
        self.samples = samples
        self.shape = shape
        self._shape = (int(np.sqrt(shape[0])), int(np.sqrt(shape[0])), shape[1])
        self._mask = convert_locations_to_mask(self.samples, self._shape)

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 2-D image.

        Parameters
        ----------
        img: np.ndarray
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        return np.reshape(self._mask * pfft.fft2(np.reshape(img, self._shape), axes=(0, 1)), self.shape)

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 2-D + T
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: np.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        return np.reshape(pfft.ifft2(np.reshape(x, self._shape) * self._mask, axes=(0, 1)), self.shape)


# class FFT2TMultiScale(FourierBase):
#     def __init__(self, samples, shape, multi_scale_factor=1):
#         self.samples = samples
#         self.shape = shape
#         self.msf = multi_scale_factor
#         self._shape = (int(np.sqrt(self.shape[0])), int(np.sqrt(self.shape[0])), self.shape[1], self.msf)
#         self._mask = convert_locations_to_mask(self.samples, self._shape)
#
#     def op(self, img):
#         return np.reshape(self._mask * pfft.fft2(np.reshape(np.sum(img, axis=-1),
#                                                             self._shape), axes=(0, 1)), self.shape)
#
#     def adj_op(self, x):
#         res = np.reshape(pfft.ifft2(np.reshape(x, self._shape) * self._mask, axes=(0, 1)), self.shape)
#         return np.repeat(res[:, :, np.newaxis], self.msf, axis=2)
