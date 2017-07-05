##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains linears operators classes.
"""

# System import
import numpy

# Package import
import pisap.extensions.transform
from pisap.base.transform import WaveletTransformBase


class Wavelet(object):
    """ This class defines the wavelet transform operators.
    """
    def __init__(self, wavelet, nb_scale=4):
        """ Initialize the Wavelet class.

        Parameters
        ----------
        wavelet: str
            the wavelet to be used during the decomposition.
        nb_scales: int, default 4
            The number of scales in the decomposition.
        """
        self.nb_scale = nb_scale
        if wavelet not in WaveletTransformBase.REGISTRY:
            raise ValueError("Unknown tranformation '{0}'.".format(wavelet))
        self.transform = WaveletTransformBase.REGISTRY[wavelet](
            nb_scale=nb_scale, verbose=0)

    def op(self, data):
        """ Operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray
            Input data array, a 2D image.

        Returns
        -------
        coeffs: ndarray
            The wavelet coefficients.
        """
        self.transform.data = data
        self.transform.analysis()
        return self.transform.analysis_data

    def adj_op(self, coeffs, dtype="array"):
        """ Adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            The wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pisap.Image.

        Returns
        -------
        ndarray reconstructed data.
        """
        self.transform.analysis_data = coeffs
        image = self.transform.synthesis()
        if dtype == "array":
            return image.data
        return image

    def l2norm(self, data_shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        data_shape: uplet
            the data shape.
        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data.
        data_shape = numpy.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = numpy.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)
