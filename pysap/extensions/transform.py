# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Wavelet transform module.

Available transform from ISAP are:

- LinearWaveletTransformATrousAlgorithm
- BsplineWaveletTransformATrousAlgorithm
- WaveletTransformInFourierSpace
- MorphologicalMedianTransform
- MorphologicalMinmaxTransform
- PyramidalLinearWaveletTransform
- PyramidalBsplineWaveletTransform
- PyramidalWaveletTransformInFourierSpaceAlgo1
- MeyerWaveletsCompactInFourierSpace
- PyramidalMedianTransform
- PyramidalLaplacian
- MorphologicalPyramidalMinmaxTransform
- DecompositionOnScalingFunction
- MallatWaveletTransform79Filters
- FeauveauWaveletTransform
- FeauveauWaveletTransformWithoutUndersampling
- LineColumnWaveletTransform1D1D
- HaarWaveletTransform
- HalfPyramidalTransform
- MixedHalfPyramidalWTAndMedianMethod
- UndecimatedDiadicWaveletTransform
- MixedWTAndPMTMethod
- UndecimatedHaarTransformATrousAlgorithm
- UndecimatedBiOrthogonalTransform
- NonOrthogonalUndecimatedTransform
- IsotropicAndCompactSupportWaveletInFourierSpace
- PyramidalWaveletTransformInFourierSpaceAlgo2
- FastCurveletTransform
- WaveletTransformViaLiftingScheme
- OnLine53AndOnColumn44
- OnLine44AndOnColumn53
"""

# System import
from __future__ import print_function, absolute_import

# Package import
from pysap.base.transform import WaveletTransformBase
from pysap.extensions import ISAP_FLATTEN
from pysap.extensions import ISAP_UNFLATTEN

# Third party import
import numpy


class ISAPWaveletTransformBase(WaveletTransformBase):
    """ Define the structure that will be used to store the ISAP results.
    """
    __isap_transform_id__ = None
    __isap_name__ = None
    __is_decimated__ = None
    __isap_nb_bands__ = None
    __isap_scale_shift__ = 0

    def _set_transformation_parameters(self):
        """ Declare transformation parameters.
        """
        # Check transformation has been defined
        if (self.__isap_transform_id__ is None or self.__isap_name__ is None or
                self.__is_decimated__ is None or
                self.__isap_nb_bands__ is None):
            raise ValueError("ISAPWaveletTransform is not defined properly.")
        self.name = self.__isap_name__

        # Get transformation parameters
        if self.__is_decimated__:
            params = ISAPWaveletTransformBase.decimated(
                self.nb_scale, self._iso_shape, self.__isap_nb_bands__,
                self.__isap_scale_shift__)
        else:
            params = ISAPWaveletTransformBase.undecimated(
                self.nb_scale, self._iso_shape, self.__isap_nb_bands__)
        (self.bands_names, self.flatten_fct, self.unflatten_fct,
         self.is_decimated, self.nb_band_per_scale, self.bands_lengths,
         self.bands_shapes) = params
        self.isap_transform_id = self.__isap_transform_id__

        # Update the default parameters
        self._update_default_transformation_parameters()

    def _update_default_transformation_parameters(self):
        """ Add a method to tune the default transformation parameters.
        """
        pass

    @classmethod
    def undecimated(cls, nb_scale, iso_shape, nb_band):
        """ Compute undecimated transformation parameters.

        Parameters
        ----------
        nb_scale: int
            the number of scale of the decomposition that includes the
            approximation scale.
        iso_shape: int
            the data isotropic shape.
        nb_band: int
            the number of band.

        Returns
        -------
        bands_names: list of str
            the name of the different bands.
        flatten_fct: int
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
        """
        if nb_band == 1:
            bands_names = ["a"]
        elif nb_band == 2:
            bands_names = ["d1", "d2"]
        elif nb_band == 3:
            bands_names = ["v", "d", "h"]
        else:
            raise ValueError("'{0} bands not yet accepted.".format(nb_band))
        flatten_fct = ISAP_FLATTEN[0]
        unflatten_fct = ISAP_UNFLATTEN[0]
        is_decimated = False
        nb_band_per_scale = numpy.ones((nb_scale, 1), dtype=int)
        nb_band_per_scale[:-1] = nb_band
        bands_lengths = (
            (iso_shape * iso_shape) *
            numpy.ones((nb_scale, nb_band), dtype=int))
        bands_shapes = WaveletTransformBase.bands_shapes(bands_lengths)

        return (bands_names, flatten_fct, unflatten_fct, is_decimated,
                nb_band_per_scale, bands_lengths, bands_shapes)

    @classmethod
    def decimated(cls, nb_scale, iso_shape, nb_band, scale_shift=0):
        """ Compute decimated transformation parameters.

        Parameters
        ----------
        nb_scale: int
            the number of scale of the decomposition that includes the
            approximation scale.
        iso_shape: int
            the data isotropic shape.
        nb_band: int
            the number of band.
        scale_shift: int, default 0
            decimate the image with a factor of 2**(scale + scale_shift).

        Returns
        -------
        bands_names: list of str
            the name of the different bands.
        flatten_fct: int
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
        """
        if nb_band == 1:
            bands_names = ["a"]
            flatten_fct = ISAP_FLATTEN[1]
            unflatten_fct = ISAP_UNFLATTEN[1]
        elif nb_band == 2:
            bands_names = ["d1", "d2"]
            flatten_fct = ISAP_FLATTEN[4]
            unflatten_fct = ISAP_UNFLATTEN[4]
        elif nb_band == 3:
            bands_names = ["v", "d", "h"]
            flatten_fct = ISAP_FLATTEN[2]
            unflatten_fct = ISAP_UNFLATTEN[2]
        else:
            raise ValueError("'{0} bands not yet accepted.".format(nb_band))
        is_decimated = True
        nb_band_per_scale = numpy.ones((nb_scale, 1), dtype=int)
        nb_band_per_scale[:-1] = nb_band
        bands_lengths = (
            iso_shape * numpy.ones((nb_scale, nb_band), dtype=int))
        bands_lengths[-1, 1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**(i + scale_shift)
        bands_lengths[-1, :] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        bands_shapes = WaveletTransformBase.bands_shapes(bands_lengths)

        return (bands_names, flatten_fct, unflatten_fct, is_decimated,
                nb_band_per_scale, bands_lengths, bands_shapes)


class LinearWaveletTransformATrousAlgorithm(ISAPWaveletTransformBase):
    """ Linear wavelet transform: a trous algorithm.
    """
    __isap_transform_id__ = 1
    __isap_name__ = "linear wavelet transform: a trous algorithm"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class BsplineWaveletTransformATrousAlgorithm(ISAPWaveletTransformBase):
    """ Bspline wavelet transform: a trous algorithm.
    """
    __isap_transform_id__ = 2
    __isap_name__ = "linear wavelet transform: a trous algorithm"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class WaveletTransformInFourierSpace(ISAPWaveletTransformBase):
    """ Wavelet transform in Fourier space.
    """
    __isap_transform_id__ = 3
    __isap_name__ = "wavelet transform in Fourier space"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class MorphologicalMedianTransform(ISAPWaveletTransformBase):
    """ Morphological median transform.
    """
    __isap_transform_id__ = 4
    __isap_name__ = "morphological median transform"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class MorphologicalMinmaxTransform(ISAPWaveletTransformBase):
    """ Morphological minmax transform.
    """
    __isap_transform_id__ = 5
    __isap_name__ = "morphological minmax transform"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class PyramidalLinearWaveletTransform(ISAPWaveletTransformBase):
    """ Pyramidal linear wavelet transform.
    """
    __isap_transform_id__ = 6
    __isap_name__ = "pyramidal linear wavelet transform"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class PyramidalBsplineWaveletTransform(ISAPWaveletTransformBase):
    """ Pyramidal bspline wavelet transform.
    """
    __isap_transform_id__ = 7
    __isap_name__ = "pyramidal bspline wavelet transform"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class PyramidalWaveletTransformInFourierSpaceAlgo1(ISAPWaveletTransformBase):
    """ Pyramidal wavelet transform in Fourier space: algo 1
    (diff. between two resolutions).
    """
    __isap_transform_id__ = 8
    __isap_name__ = ("pyramidal wavelet transform in Fourier space: algo 1 "
                     "(diff. between two resolutions)")
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class MeyerWaveletsCompactInFourierSpace(ISAPWaveletTransformBase):
    """ Meyers wavelets (compact support in Fourier space).
    """
    __isap_transform_id__ = 9
    __isap_name__ = "Meyers wavelets (compact support in Fourier space)"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class PyramidalMedianTransform(ISAPWaveletTransformBase):
    """ Pyramidal median transform (PMT).
    """
    __isap_transform_id__ = 10
    __isap_name__ = "pyramidal median transform (PMT)"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class PyramidalLaplacian(ISAPWaveletTransformBase):
    """ Pyramidal laplacian.
    """
    __isap_transform_id__ = 11
    __isap_name__ = "pyramidal laplacian"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class MorphologicalPyramidalMinmaxTransform(ISAPWaveletTransformBase):
    """ Morphological pyramidal minmax transform.
    """
    __isap_transform_id__ = 12
    __isap_name__ = "morphological pyramidal minmax transform"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class DecompositionOnScalingFunction(ISAPWaveletTransformBase):
    """ Decomposition on scaling function.
    """
    __isap_transform_id__ = 13
    __isap_name__ = "decomposition on scaling function"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class MallatWaveletTransform79Filters(ISAPWaveletTransformBase):
    """ Mallat's wavelet transform (7/9 filters).
    """
    __isap_transform_id__ = 14
    __isap_name__ = "Mallat's wavelet transform (7/9 filters)"
    __is_decimated__ = True
    __isap_nb_bands__ = 3
    __isap_scale_shift__ = 1


class FeauveauWaveletTransform(ISAPWaveletTransformBase):
    """ Feauveau's wavelet transform.
    """
    __isap_transform_id__ = 15
    __isap_name__ = "Feauveau's wavelet transform"
    __is_decimated__ = True
    __isap_nb_bands__ = 2
    __isap_scale_shift__ = 1

    def _set_transformation_parameters(self):
        raise NotImplementedError(
            "This transformation is not yet accessible from the wrapping, "
            "please use the Python bindings.")
        self.name = "Feauveau's wavelet transform"
        ratios = numpy.ones_like(bands_lengths, dtype=float)
        ratios[:, 1] *= 2.0
        params = decimated(self.nb_scale, self._iso_shape, 2,
                           scale_shift=1)
        (self.bands_names, self.flatten_fct, self.unflatten_fct,
         self.is_decimated, self.nb_band_per_scale, self.bands_lengths,
         self.bands_shapes) = params
        self.isap_transform_id = 15


class FeauveauWaveletTransformWithoutUndersampling(ISAPWaveletTransformBase):
    """ Feauveau's wavelet transform without undersampling.
    """
    __isap_transform_id__ = 16
    __isap_name__ = "Feauveau's wavelet transform without undersampling"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class LineColumnWaveletTransform1D1D(ISAPWaveletTransformBase):
    """ Line Column Wavelet Transform (1D+1D).
    """
    __isap_transform_id__ = 17
    __isap_name__ = "Line Column Wavelet Transform (1D+1D)"
    __is_decimated__ = False
    __isap_nb_bands__ = 1

    def _set_transformation_parameters(self):
        raise NotImplementedError(
            "This transformation is not yet accessible from the wrapping, "
            "please use the Python bindings.")
        _map = {128: 5, 512: 6}
        self.nb_scale = _map[self._iso_shape]  # fixed for this wavelet
        self.name = "Line Column Wavelet Transform (1D+1D)"
        self.bands_names = ['d%d' % i
                            for i in range(_map[self._iso_shape])]
        self.nb_band_per_scale = numpy.array([
            _map[self._iso_shape]] * nb_scale)
        self.bands_lengths = (
            (self._iso_shape * self._iso_shape) *
            numpy.ones((nb_scale,  _map[self._iso_shape]), dtype=int))
        self.bands_shapes = WaveletTransformBase.bands_shapes(
            bands_lengths)
        self.isap_transform_id = 17
        self.flatten_fct = ISAP_FLATTEN[0]
        self.is_decimated = False


class HaarWaveletTransform(ISAPWaveletTransformBase):
    """ Haar's wavelet transform.
    """
    __isap_transform_id__ = 18
    __isap_name__ = "linear wavelet transform: a trous algorithm"
    __is_decimated__ = True
    __isap_nb_bands__ = 3
    __isap_scale_shift__ = 1


class HalfPyramidalTransform(ISAPWaveletTransformBase):
    """ Half-pyramidal transform.
    """
    __isap_transform_id__ = 19
    __isap_name__ = "half-pyramidal transform"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class MixedHalfPyramidalWTAndMedianMethod(ISAPWaveletTransformBase):
    """ Mixed Half-pyramidal WT and Median method (WT-HPMT).
    """
    __isap_transform_id__ = 20
    __isap_name__ = "mixed Half-pyramidal WT and Median method (WT-HPMT)"
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class UndecimatedDiadicWaveletTransform(ISAPWaveletTransformBase):
    """ Undecimated diadic wavelet transform (two bands per scale).
    """
    __isap_transform_id__ = 21
    __isap_name__ = ("undecimated diadic wavelet transform (two bands per "
                     "scale)")
    __is_decimated__ = False
    __isap_nb_bands__ = 2

    def _update_default_transformation_parameters(self):
        self.bands_lengths[-1, 1:] = 0


class MixedWTAndPMTMethod(ISAPWaveletTransformBase):
    """ Mixed WT and PMT method (WT-PMT).
    """
    __isap_transform_id__ = 22
    __isap_name__ = "mixed WT and PMT method (WT-PMT)"
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class UndecimatedHaarTransformATrousAlgorithm(ISAPWaveletTransformBase):
    """ Undecimated Haar transform: a trous algorithm (one band per scale).
    """
    __isap_transform_id__ = 23
    __isap_name__ = ("undecimated Haar transform: a trous algorithm "
                     "(one band per scale)")
    __is_decimated__ = False
    __isap_nb_bands__ = 1


class UndecimatedBiOrthogonalTransform(ISAPWaveletTransformBase):
    """ Undecimated (bi-) orthogonal transform (three bands per scale.
    """
    __isap_transform_id__ = 24
    __isap_name__ = ("undecimated (bi-) orthogonal transform (three bands "
                     "per scale")
    __is_decimated__ = False
    __isap_nb_bands__ = 3

    def _update_default_transformation_parameters(self):
        self.bands_lengths[-1, 1] = 0.
        self.bands_lengths[-1, 2] = 0.


class NonOrthogonalUndecimatedTransform(ISAPWaveletTransformBase):
    """ Non orthogonal undecimated transform (three bands per scale).
    """
    __isap_transform_id__ = 25
    __isap_name__ = ("non orthogonal undecimated transform (three bands per "
                     "scale)")
    __is_decimated__ = False
    __isap_nb_bands__ = 3

    def _update_default_transformation_parameters(self):
        self.bands_lengths[-1, 1] = 0.
        self.bands_lengths[-1, 2] = 0.


class IsotropicAndCompactSupportWaveletInFourierSpace(
        ISAPWaveletTransformBase):
    """ Isotropic and compact support wavelet in Fourier space.
    """
    __isap_transform_id__ = 26
    __isap_name__ = ("Isotropic and compact support wavelet in Fourier "
                     "space")
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class PyramidalWaveletTransformInFourierSpaceAlgo2(ISAPWaveletTransformBase):
    """ Pyramidal wavelet transform in Fourier space: algo 2
    (diff. between the square of two resolutions).
    """
    __isap_transform_id__ = 27
    __isap_name__ = ("pyramidal wavelet transform in Fourier space: algo 2 "
                     "(diff. between the square of two resolutions)")
    __is_decimated__ = True
    __isap_nb_bands__ = 1


class FastCurveletTransform(ISAPWaveletTransformBase):
    """ Fast Curvelet Transform.
    """
    __isap_transform_id__ = 28
    __isap_name__ = "Fast Curvelet Transform"
    __is_decimated__ = False
    __isap_nb_bands__ = 1

    def _set_transformation_parameters(self):
        raise NotImplementedError(
            "This transformation is not yet accessible from the wrapping, "
            "please use the Python bindings.")
        self.name = "Fast Curvelet Transform"
        self.bands_names = ["d"] * 16
        self.nb_band_per_scale = [16, 16, 8, 8, 8, 8, 8, 8, 8, 1]
        self.nb_band_per_scale = numpy.array(
            nb_band_per_scale[:self.nb_scale])
        self.nb_band_per_scale[-1] = 1
        self.bands_shapes = get_curvelet_bands_shapes(
            self.data.shape, nb_scale, nb_band_per_scale)
        if nb_scale == 2:
            self.bands_shapes[-1] = [
                (bands_shapes[0][0][0], bands_shapes[0][0][0])]
        else:
            self.bands_shapes[-1] = [
                (bands_shapes[-1][0][0], bands_shapes[-1][0][0])]
        self.bands_lengths = numpy.zeros(
            (nb_scale, nb_band_per_scale.max()), dtype=int)
        for ks in range(nb_scale):
            for kb in range(nb_band_per_scale[ks]):
                self.bands_lengths[ks, kb] = (
                    bands_shapes[ks][kb][0] * bands_shapes[ks][kb][1])
        self.isap_transform_id = 28
        self.flatten_fct = ISAP_FLATTEN[3]
        self.is_decimated = False  # since it's a not an 2**i decimation...


class WaveletTransformViaLiftingScheme(ISAPWaveletTransformBase):
    """ Wavelet transform via lifting scheme.
    """
    __isap_transform_id__ = 29
    __isap_name__ = "Wavelet transform via lifting scheme"
    __is_decimated__ = True
    __isap_nb_bands__ = 3
    __isap_scale_shift__ = 1


class OnLine53AndOnColumn44(ISAPWaveletTransformBase):
    """ 5/3 on line and 4/4 on column.
    """
    __isap_transform_id__ = 30
    __isap_name__ = "5/3 on line and 4/4 on column"
    __is_decimated__ = False
    __isap_nb_bands__ = 3

    def _update_default_transformation_parameters(self):
        self.bands_names = ["a", "a", "a"]
        self.bands_lengths[-1, 1:] = 0


class OnLine44AndOnColumn53(ISAPWaveletTransformBase):
    """ 4/4 on line and 5/3 on column.
    """
    __isap_transform_id__ = 31
    __isap_name__ = "4/4 on line and 5/3 on column"
    __is_decimated__ = False
    __isap_nb_bands__ = 3

    def _update_default_transformation_parameters(self):
        self.bands_names = ["a", "a", "a"]
        self.bands_lengths[-1, 1:] = 0
