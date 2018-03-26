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


# Package import
import pysap
from .utils import flatten_swtn
from .utils import unflatten_swtn
from .utils import flatten_wave
from .utils import unflatten_wave

# Third party import
import numpy
import pywt


class pyWavelet3(object):
    """ The 3D wavelet transform class from pyWavelets package.
    """
    def __init__(self, wavelet_name, nb_scale=4, verbose=0, undecimated=False):
        """ Initialize the 'pyWavelet3' class.
            print(x_new.shape)
        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        verbose: int, default 0
            the verbosity level.
        undecimated: bool, default False
            enable use undecimated wavelet transform.
        """
        self.nb_scale = nb_scale
        if wavelet_name not in pywt.wavelist():
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        self.transform = pywt.Wavelet(wavelet_name)
        self.nb_scale = nb_scale-1
        self.undecimated = undecimated
        self.coeffs_shape = None

    def op(self, data):
        """ Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 3D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        if isinstance(data, numpy.ndarray):
            data = pysap.Image(data=data)
        if self.undecimated:
            coeffs_dict = pywt.swtn(data, self.transform, level=self.nb_scale)
            coeffs, self.coeffs_shape = flatten_swtn(coeffs_dict)
            return coeffs
        else:
            coeffs_dict = pywt.wavedecn(data,
                                        self.transform,
                                        level=self.nb_scale)
            coeffs, self.coeffs_shape = flatten_wave(coeffs_dict)
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
            pysap.Image.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.undecimated:
            coeffs_dict = unflatten_swtn(coeffs, self.coeffs_shape)
            data = pywt.iswtn(coeffs_dict,
                              self.transform)
        else:
            coeffs_dict = unflatten_wave(coeffs, self.coeffs_shape)
            data = pywt.waverecn(
                coeffs=coeffs_dict,
                wavelet=self.transform)
        if dtype == "array":
            return data
        return pysap.Image(data=data)

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
        print(shape)
        shape = numpy.asarray(shape)
        shape += shape % 2
        fake_data = numpy.zeros(shape)
        fake_data[zip(shape / 2)] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)
