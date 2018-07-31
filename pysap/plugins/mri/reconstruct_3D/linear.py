# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
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
        if wavelet_name not in pywt.wavelist():
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        self.pywt_transform = pywt.Wavelet(wavelet_name)
        self.nb_scale = nb_scale-1
        self.undecimated = undecimated
        self.unflatten = unflatten_swtn if undecimated else unflatten_wave
        self.flatten = flatten_swtn if undecimated else flatten_wave
        self.coeffs_shape = None

    def get_coeff(self):
        """ Return the wavelet coeffiscients
        Return:
        -------
        The wavelet coeffiscients value
        """
        return self.coeffs

    def set_coeff(self, coeffs):
        """ Set the wavelet coefficients value
        """
        self.coeffs = coeffs  # XXX: TODO: add some checks

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
            coeffs_dict = pywt.swtn(data, self.pywt_transform,
                                    level=self.nb_scale)
            coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
            return coeffs
        else:
            coeffs_dict = pywt.wavedecn(data,
                                        self.pywt_transform,
                                        level=self.nb_scale)
            self.coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
            return self.coeffs

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
        self.coeffs = coeffs
        if self.undecimated:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.iswtn(coeffs_dict,
                              self.pywt_transform)
        else:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.waverecn(
                coeffs=coeffs_dict,
                wavelet=self.pywt_transform)
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

        fake_data[[(int(i[0]),) for i in list(zip(shape/2))]] = 1
        # WARNING: this line is overly complicated, but it basically does this:
        # fake_data[zip(shape / 2)] = 1
        # It is written as such to help Python2.x/3.x compatibility

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)
