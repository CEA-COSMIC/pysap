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
This module contains linear operator classes.
"""

# System import
import numpy as np

# Package import
from pisap import WaveletTransform


class Wavelet(object):
    """ Wavelet class.

    This class defines the wavelet transform operators
    """

    def __init__(self, maxscale=4, **kwargs):
        """ Initialize the Wavelet class.
        """
        self.maxscale = maxscale
        self.isap_kwargs = kwargs

    def op(self, data):
        """ Operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image.

        Returns
        -------
        np.ndarray wavelet convolved data.
        """
        trf = WaveletTransform(data=data, wavelet="",
                               maxscale=self.maxscale, use_isap=True)
        trf.analysis(**self.isap_kwargs)
        return trf

    def adj_op(self, trf, dtype="array"):
        """ Adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        trf : WaveletTransform
            wavelet coefficients store in a wavelet tree.
        dtype: str (optional, default 'array')
            if 'array' return the data as a ndarray, otherwise return an image.

        Returns
        -------
        np.ndarray reconstructed data.
        """
        image = trf.synthesis()
        if dtype == "array":
            return image.data
        return image

    def l2norm(self, data_shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        data_shape: uplet
            the ndarray data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data.
        data_shape = np.asarray(data_shape)
        data_shape += data_shape % 2 - 1
        fake_data = np.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        mr_filters = self.op(fake_data).to_cube()

        # Compute the L1 norm
        norm = 0
        for fltr in mr_filters[:, 0]:
            norm +=  np.linalg.norm(fltr)

        return norm



class Identity():
    """ Identity operator class
    This is a dummy class that can be used in the optimisation classes
    """

    def __init__(self):
        self.l1norm = 1.0

    def op(self, data, **kwargs):
        """ Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        **kwargs
            Arbitrary keyword arguments
        Returns
        -------
        np.ndarray input data
        """
        return data

    def adj_op(self, data):
        """ Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        Returns
        -------
        np.ndarray input data
        """
        return data

