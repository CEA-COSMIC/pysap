"""
Defines linear operators for fMRI
"""

#Package imports
from builtins import zip
from .transform import FTransform

# Third party import
import numpy as np


class Wavelet2T(object):
    """
    Wavelet transform for 2D+T data
    """
    def __init__(self, wavelet_name, nb_scale=4, wavelet_name_t=None, nb_scale_t=1, verbose=0):
        self.nb_scale = nb_scale
        self.nb_scale_t = nb_scale_t
        self.transform = FTransform(wavelet_name, nb_scale, wavelet_name_t, nb_scale_t, verbose)
        self.coeffs_shape = None

    def op(self, data):
        """ Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.
        The convolution is done frame by frame

        Parameters
        ----------
        data: ndarray or Image
            input 3D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        coeffs, self.coeffs_shape = self.transform.analysis(data)
        return coeffs

    def adj_op(self, coeffs, dtype="array"):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pisap.Image.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        image = []
        # for i in range(coeffs.shape[-1]):
        #     self.transform.analysis_data = unflatten(coeffs[:, i], self.coeffs_shape)
        #     image.append(self.transform.synthesis())
        # if dtype == "array":
        #     image = np.moveaxis([image_.data for image_ in image], 0, -1)
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


