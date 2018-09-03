# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""

# System import
import warnings
import numpy as np

# Package import
from .utils import convert_locations_to_mask
from .utils import normalize_frequency_locations

# Third party import
try:
    import pynfft
except Exception:
    warnings.warn("pynfft python package has not been found. If needed use "
                  "the master release.")
    pass
import scipy.fftpack as pfft


class FourierBase(object):
    """ Base Fourier transform operator class.
    """
    def op(self, img):
        """ This method calculates Fourier transform.

        Parameters
        ----------
        img: np.ndarray
            input image as array.

        Returns
        -------
        result: np.ndarray
            Fourier transform of the image.
        """
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, x):
        """ This method calculates inverse Fourier transform of real or complex
        sequence.

        Parameters
        ----------
        x: np.ndarray
            input Fourier data array.

        Returns
        -------
        results: np.ndarray
            inverse discrete Fourier transform.
        """
        raise NotImplementedError("'adj_op' is an abstract method.")


class FFT2(FourierBase):
    """ Standard 2D Fast Fourrier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """
    def __init__(self, samples, shape):
        """ Initilize the 'FFT2' class.

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
        return self._mask * pfft.fft2(img)

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 2-D
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
        return pfft.ifft2(self._mask * x)


class NFFT(FourierBase):
    """ ND non catesian Fast Fourrier Transform class
    The NFFT will normalize in a symmetric way the direct and adjoint operator.
    This means that both the direct and adjoint operator will be divided by the
    squarre root of the number of samples in the fourier domain.

    Attributes
    ----------
    samples: np.ndarray
        the samples locations in the Fourier domain between [-0.5; 0.5[.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """

    def __init__(self, samples, shape):
        """ Initilize the 'NFFT' class.

        Parameters
        ----------
        samples: np.ndarray (Mxd)
            the samples locations in the Fourier domain where M is the number
            of samples and d is the dimensionnality of the output data
            (2D for an image, 3D for a volume).
        shape: tuple of int
            shape of the image (not necessarly a square matrix).

        Exemple
        -------
        >>> import numpy as np
        >>> from pysap.data import get_sample_data
        >>> from pysap.numerics.fourier import NFFT
        >>> from pysap.plugins.mri.reconstruct.utils import \
        convert_mask_to_locations

        >>> I = get_sample_data("2d-pmri").data.astype("complex128")
        >>> I = I[0]
        >>> samples = convert_mask_to_locations(np.ones(I.shape))
        >>> fourier_op = NFFT(samples=samples, shape=I.shape)
        >>> x_nfft = fourier_op.op(I)
        >>> x_fft = np.fft.ifftshift(np.fft.fft2(np.fftshift(I))).flatten()
        >>> np.mean(np.abs(x_fft / np.sqrt(np.prod(I.shape)) / x_nfft))
        1.000000000000005
        """
        if samples.shape[-1] != len(shape):
            raise ValueError("Samples and Shape dimension doesn't correspond")
        self.samples = samples
        if samples.min() < -0.5 or samples.max() >= 0.5:
            warnings.warn("Samples will be normalized between [-0.5; 0.5[")
            self.samples = normalize_frequency_locations(self.samples)
        self.plan = pynfft.NFFT(N=shape, M=len(samples))
        self.shape = shape
        self.plan.x = self.samples
        self.plan.precompute()

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a N-D data.

        Parameters
        ----------
        img: np.ndarray
            input ND array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        self.plan.f_hat = img
        return np.sqrt(1.0 / self.plan.M) * np.copy(self.plan.trafo())

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
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        self.plan.f = x
        return np.sqrt(1.0 / self.plan.M) * np.copy(self.plan.adjoint())
