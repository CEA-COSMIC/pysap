# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Wavelet transform base module.
"""

# System import
from pprint import pprint
import uuid
import os
import warnings

# Package import
import pysap
from .utils import with_metaclass
from pysap.plotting import plot_transform

# Third party import
import numpy


class MetaRegister(type):
    """ Simple Python metaclass registry pattern.
    """
    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        """ Allocation.

        Parameters
        ----------
        name: str
            the name of the class.
        bases: tuple
            the base classes.
        attrs:
            the attributes defined for the class.
        """
        new_cls = type.__new__(cls, name, bases, attrs)
        if name in cls.REGISTRY:
            raise ValueError(
                "'{0}' name already used in registry.".format(name))
        if name not in ("WaveletTransformBase", "ISAPWaveletTransformBase",
                        "PyWaveletTransformBase"):
            cls.REGISTRY[name] = new_cls
        return new_cls


class WaveletTransformBase(with_metaclass(MetaRegister)):
    """ Data structure representing a signal wavelet decomposition.

    Available transforms are define in 'pysap.transform'.
    """
    def __init__(self, nb_scale, verbose=0, dim=2, use_wrapping=False,
                 **kwargs):
        """ Initialize the WaveletTransformBase class.

        Parameters
        ----------
        data: ndarray
            the input data.
        nb_scale: int
            the number of scale of the decomposition that includes the
            approximation scale.
        verbose: int, default 0
            control the verbosity level.
        dim: int, default 2
            define the data dimension.
        use_wrapping: bool, default False
            if set, in the case of ISAP, use the command lines rather than the
            bindings.
        """
        # Wavelet transform parameters
        self.nb_scale = nb_scale
        self.name = None
        self.bands_names = None
        self.nb_band_per_scale = None
        self.is_decimated = None
        self.data_dim = dim
        self.use_wrapping = use_wrapping

        # Data that can be decalred afterward
        self._data = None
        self._image_metadata = {}
        self._data_shape = None
        self._iso_shape = None
        self._analysis_data = None
        self._analysis_shape = None
        self._analysis_header = None
        self._analysis_buffer_shape = None
        self.verbose = verbose

        self.kwargs = kwargs

        # Transformation
        self._init_transform(**self.kwargs)

    def __reduce__(self):
        """ The interface to pickle dump call.

        Return
        ------
        reduced_instance: tuple,
            two or five items long tuple to define the init of a pickled
            instance.
        """
        return (self.__class__, (self.nb_scale, self.verbose))

    def __getitem__(self, given):
        """ Access the analysis designated scale/band coefficients.

        Parameters
        ----------
        given: int, slice or tuple
            the scale and band indices.

        Returns
        -------
        coeffs: ndarray or list of ndarray
            the analysis coefficients.
        """
        # Convert given index to generic scale/band index
        if not isinstance(given, tuple):
            given = (given, slice(None))

        # Check that we have a valid given index
        if len(given) != 2:
            raise ValueError("Expect a scale/band int or 2-uplet index.")

        # Check some data are stored in the structure
        if self._analysis_data is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients array.")

        # Handle multi-dim slice object
        if isinstance(given[0], slice):
            start = given[0].start or 0
            stop = given[0].stop or self.nb_scale
            step = given[0].step or 1
            coeffs = [self.__getitem__((index, given[1]))
                      for index in range(start, stop, step)]
        elif isinstance(given[1], slice):
            start = given[1].start or 0
            stop = given[1].stop or self.nb_band_per_scale[given[0]]
            step = given[1].step or 1
            coeffs = [self.band_at(given[0], index)
                      for index in range(start, stop, step)]
        else:
            coeffs = [self.band_at(given[0], given[1])]

        # Format output
        if len(coeffs) == 1:
            coeffs = coeffs[0]

        return coeffs

    def __setitem__(self, given, array):
        """ Set the analysis designated scale/band coefficients.

        Parameters
        ----------
        given: tuple
            the scale and band indices.
        array: ndarray
            the specific scale/band data as an array.
        """
        # Check that we have a valid given index
        if len(given) != 2:
            raise ValueError("Expect a scale/band int or 2-uplet index.")

        # Check given index
        if isinstance(given[0], slice) or isinstance(given[1], slice):
            raise ValueError("Expect a scale/band int index (no slice).")

        # Check some data are stored in the structure
        if self._analysis_data is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients array.")

        # Handle multidim slice object
        if isinstance(given[0], slice):
            start = given[0].start or 0
            stop = given[0].stop or self.nb_scale
            step = given[0].step or 1
            coeffs = [self.__getitem__((index, given[1]))
                      for index in range(start, stop, step)]
        elif isinstance(given[1], slice):
            start = given[1].start or 0
            stop = given[1].stop or self.nb_band_per_scale[given[0]]
            step = given[1].step or 1
            coeffs = [self.band_at(given[0], index)
                      for index in range(start, stop, step)]
        else:
            coeffs = [self.band_at(given[0], given[1])]

        # Format output
        if len(coeffs) == 1:
            coeffs = coeffs[0]

        return coeffs

    ##########################################################################
    # Properties
    ##########################################################################

    def _set_data(self, data):
        """ Set the input data array.

        Parameters
        ----------
        data: nd-array or pysap.Image
            input data/signal.
        """
        if self.verbose > 0 and self._data is not None:
            print("[info] Replacing existing input data array.")
        # Ensure that the shape is square except when the family is pywt
        if self.__family__ != 'pywt' and \
                not all([e == data.shape[0] for e in data.shape]):
            raise ValueError("Expect a square shape data.")
        if data.ndim != self.data_dim:
            raise ValueError("This wavelet can only be applied on {0}D "
                             "square images".format(self.data_dim))
        if self.is_decimated and not (data.shape[0] // 2**(self.nb_scale) > 0):
            raise ValueError("Can't decimate the data with the specified "
                             "number of scales.")
        if isinstance(data, pysap.Image):
            self._data = data.data
            self._image_metadata = data.metadata
        else:
            self._data = data
        self._data_shape = self._data.shape
        self._iso_shape = self._data_shape[0]

        if self.use_wrapping:
            self._set_transformation_parameters()
            self._compute_transformation_parameters()

    def _get_data(self):
        """ Get the input data array.

        Returns
        -------
        data: nd-array
            input data/signal.
        """
        return self._data

    def _set_analysis_data(self, analysis_data):
        """ Set the decomposition coefficients array.

        Parameters
        ----------
        analysis_data: lsit of nd-array
            decomposition coefficients array.
        """
        if self.verbose > 0 and self._analysis_data is not None:
            print("[info] Replacing existing decomposition coefficients "
                  "array.")
        if len(analysis_data) != sum(self.nb_band_per_scale):
            raise ValueError("The wavelet coefficients do not correspond to "
                             "the wavelet transform parameters.")
        self._analysis_data = analysis_data

    def _get_analysis_data(self):
        """ Get the decomposition coefficients array.

        Returns
        -------
        analysis_data: nd-array
            decomposition coefficients array.
        """
        return self._analysis_data

    def _set_analysis_header(self, analysis_header):
        """ Set the decomposition coefficients header.

        Parameters
        ----------
        analysis_header: dict
            decomposition coefficients array.
        """
        if self.verbose > 0 and self._analysis_header is not None:
            print("[info] Replacing existing decomposition coefficients "
                  "header.")
        self._analysis_header = analysis_header

    def _get_analysis_header(self):
        """ Get the decomposition coefficients header.

        Returns
        -------
        analysis_header: dict
            decomposition coefficients header.
        """
        return self._analysis_header

    def _get_info(self):
        """ Return the transformation information. This iformation is only
        available when using the Python bindings.
        """
        if hasattr(self.trf, "info"):
            self.trf.info()

    data = property(_get_data, _set_data)
    analysis_data = property(_get_analysis_data, _set_analysis_data)
    analysis_header = property(_get_analysis_header, _set_analysis_header)
    info = property(_get_info)

    ##########################################################################
    # Public members
    ##########################################################################

    @classmethod
    def bands_shapes(cls, bands_lengths, ratio=None):
        """ Return the different bands associated shapes given there lengths.

        Parameters
        ----------
        bands_lengths: ndarray (<nb_scale>, max(<nb_band_per_scale>, 0))
            array holding the length between two bands of the data
            vector per scale.
        ratio: ndarray, default None
            a array containing ratios for eeach scale and each band.

        Returns
        -------
        bands_shapes: list of list of 2-uplet (<nb_scale>, <nb_band_per_scale>)
            structure holding the shape of each bands at each scale.
        """
        if ratio is None:
            ratio = numpy.ones_like(bands_lengths)
        bands_shapes = []
        for band_number, scale_data in enumerate(bands_lengths):
            scale_shapes = []
            for scale_number, scale_padd in enumerate(scale_data):
                shape = (
                    int(numpy.sqrt(
                        scale_padd * ratio[band_number, scale_number])),
                    int(numpy.sqrt(
                        scale_padd / ratio[band_number, scale_number])))
                scale_shapes.append(shape)
            bands_shapes.append(scale_shapes)
        return bands_shapes

    def show(self):
        """ Display the different bands at the different decomposition scales.
        """
        plot_transform(self)

    def analysis(self, **kwargs):
        """ Decompose a real or complex signal using ISAP.

        Fill the instance 'analysis_data' and 'analysis_header' parameters.

        Parameters
        ----------
        kwargs: dict (optional)
            the parameters that will be passed to
            'pysap.extensions.mr_tansform'.
        """
        # Checks
        if self._data is None:
            raise ValueError("Please specify first the input data.")

        # Analysis
        if numpy.iscomplexobj(self._data):
            analysis_data_real, self.analysis_header = self._analysis(
                self._data.real, **kwargs)
            analysis_data_imag, _ = self._analysis(
                self._data.imag, **kwargs)
            if isinstance(analysis_data_real, numpy.ndarray):
                self._analysis_data = (
                    analysis_data_real + 1.j * analysis_data_imag)
            else:
                self._analysis_data = [
                    re + 1.j * ima
                    for re, ima in zip(analysis_data_real, analysis_data_imag)]
        else:
            self._analysis_data, self._analysis_header = self._analysis(
                self._data, **kwargs)

    def synthesis(self):
        """ Reconstruct a real or complex signal from the wavelet coefficients
        using ISAP.

        Returns
        -------
        data: pysap.Image
            the reconstructed data/signal.
        """
        # Checks
        if self._analysis_data is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients array.")
        if self.use_wrapping and self._analysis_header is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients header.")

        # Message
        if self.verbose > 1:
            print("[info] Synthesis header:")
            pprint(self._analysis_header)

        # Reorganize the coefficents with ISAP convention
        # TODO: do not backup the list of bands
        if self.use_wrapping:
            analysis_buffer = numpy.zeros(
                self._analysis_buffer_shape, dtype=self.analysis_data[0].dtype)
            for scale, nb_bands in enumerate(self.nb_band_per_scale):
                for band in range(nb_bands):
                    self._set_linear_band(scale, band, analysis_buffer,
                                          self.band_at(scale, band))
            _saved_analysis_data = self._analysis_data
            self._analysis_data = analysis_buffer
            self._analysis_data = [self.unflatten_fct(self)]

        # Synthesis
        if numpy.iscomplexobj(self._analysis_data[0]):
            data_real = self._synthesis(
                [arr.real for arr in self._analysis_data],
                self._analysis_header)
            data_imag = self._synthesis(
                [arr.imag for arr in self._analysis_data],
                self._analysis_header)
            data = data_real + 1.j * data_imag
        else:
            data = self._synthesis(
                self._analysis_data, self._analysis_header)

        # TODO: remove this code asap
        if self.use_wrapping:
            self._analysis_data = _saved_analysis_data

        return pysap.Image(data=data, metadata=self._image_metadata)

    def band_at(self, scale, band):
        """ Get the band at a specific scale.

        Parameters
        ----------
        scale: int
            index of the scale.
        band: int
            index of the band.

        Returns
        -------
        band_data: nd-arry
            the requested band data array.
        """
        # Message
        if self.verbose > 1:
            print("[info] Accessing scale '{0}' and band '{1}'...".format(
                scale, band))

        # Get the band array
        index = numpy.sum(self.nb_band_per_scale[:scale]).astype(int) + band
        band_data = self.analysis_data[index]

        return band_data

    ##########################################################################
    # Private members
    ##########################################################################

    def _init_transform(self):
        """ Define the transform.

        Attributes
        ----------
        trf: object
            the transformation.
        """
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    def _get_linear_band(self, scale, band, analysis_data):
        """ Access the desired band data from a 1D linear analysis buffer.

        Parameters
        ----------
        scale: int
            index of the scale.
        band: int
            index of the band.
        analysis_data: nd-array (N, )
            the analysis buffer.

        Returns
        -------
        band_data: nd-arry (M, )
            the requested band buffer.
        """
        # Compute selected scale/band start/stop indices
        start_scale_padd = self.scales_padds[scale]
        start_band_padd = (
            self.bands_lengths[scale, :band + 1].sum()
            - self.bands_lengths[scale, band])
        start_padd = start_scale_padd + start_band_padd
        stop_padd = start_padd + self.bands_lengths[scale, band]

        # Get the band array
        band_data = analysis_data[start_padd: stop_padd].reshape(
            self.bands_shapes[scale][band])

        return band_data

    def _set_linear_band(self, scale, band, analysis_data, band_data):
        """ Set the desired band data in a 1D linear analysis buffer.

        Parameters
        ----------
        scale: int
            index of the scale.
        band: int
            index of the band.
        analysis_data: nd-array (N, )
            the analysis buffer.
        band_data: nd-array (M, M)
            the band data to be added in the analysis buffer.

        Returns
        -------
        analysis_data: nd-arry (N, )
            the updated analysis buffer.
        """
        # Compute selected scale/band start/stop indices
        start_scale_padd = self.scales_padds[scale]
        start_band_padd = (
            self.bands_lengths[scale, :band + 1].sum()
            - self.bands_lengths[scale, band])
        start_padd = start_scale_padd + start_band_padd
        stop_padd = start_padd + self.bands_lengths[scale, band]

        # Get the band array
        analysis_data[start_padd: stop_padd] = band_data.flatten()

        return analysis_data

    def _set_transformation_parameters(self):
        """ Define the transformation class parameters.

        Attributes
        ----------
        name: str
            the name of the decomposition.
        bands_names: list of str
            the name of the different bands.
        flatten_fct: callable
            a function used to reorganize the ISAP decomposition coefficients,
            see 'pysap/extensions/formating.py' module for more details.
        unflatten_fct: callable
            a function used to reorganize the decomposition coefficients using
            ISAP convention, see 'pysap/extensions/formating.py' module for
            more details.
        is_decimated: bool
            True if the decomposition include a decimation of the
            band number of coefficients.
        nb_band_per_scale: ndarray (<nb_scale>, )
            vector of int holding the number of band per scale.
        bands_lengths: ndarray (<nb_scale>, max(<nb_band_per_scale>, 0))
            array holding the length between two bands of the data
            vector per scale.
        bands_shapes: list of list of 2-uplet (<nb_scale>, <nb_band_per_scale>)
            structure holding the shape of each bands at each scale.
        isap_transform_id: int
            the label of the ISAP transformation.
        """
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    def _compute_transformation_parameters(self):
        """ Compute information in order to split scale/band flatten data.

        Attributes
        ----------
        scales_lengths: ndarray (<nb_scale>, )
            the length of each band.
        scales_padds: ndarray (<nb_scale> + 1, )
            the index of the data associated to each scale.
        """
        if self.bands_lengths is None:
            raise ValueError(
                "The transformation parameters have not been set.")
        self.scales_lengths = self.bands_lengths.sum(axis=1)
        self.scales_padds = numpy.zeros((self.nb_scale + 1, ), dtype=int)
        self.scales_padds[1:] = self.scales_lengths.cumsum()

    def _analysis(self, data, **kwargs):
        """ Decompose a real signal using ISAP.

        Parameters
        ----------
        data: nd-array
            a real array to be decomposed.
        kwargs: dict (optional)
            the parameters that will be passed to
            'pysap.extensions.mr_tansform'.

        Returns
        -------
        analysis_data: nd_array
            the decomposition coefficients.
        analysis_header: dict
            the decomposition associated information.
        """
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    def _synthesis(self, analysis_data, analysis_header):
        """ Reconstruct a real signal from the wavelet coefficients using ISAP.

        Parameters
        ----------
        analysis_data: list of nd-array
            the wavelet coefficients array.
        analysis_header: dict
            the wavelet decomposition parameters.

        Returns
        -------
        data: nd-array
            the reconstructed data array.
        """
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")
