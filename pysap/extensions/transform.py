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

Available 2D transform from ISAP are:

- to get the full list of builtin wavelets' names just use the pysap.wavelist
  with 'isap-2d' as the family argument.

Available 3D transform from ISAP are:

- to get the full list of builtin wavelets' names just use the pysap.wavelist
  with 'isap-3d' as the family argument.

Available transform from pywt are:

- to get the full list of builtin wavelets' names just use the pysap.wavelist
  with 'pywt' as the family argument.
"""

# System import
import os
import warnings

# Package import
import pysap
from pysap.base.transform import WaveletTransformBase
from pysap.extensions import ISAP_FLATTEN
from pysap.extensions import ISAP_UNFLATTEN
try:
    import pysparse
except ImportError:
    warnings.warn(
        'Sparse2D Python bindings not found. Any call to a Sparse2D transform '
        + 'or a plug-in method that uses a Sparse2D transform will result in '
        + 'an error.'
    )
    pysparse = None

# Third party import
import numpy
import pywt


class PyWaveletTransformBase(WaveletTransformBase):
    """ Define the structure that will be used to store the pywt results.
    """
    __family__ = "pywt"

    def __init__(self, nb_scale, verbose=0, dim=2, is_decimated=True,
                 axes=None, padding_mode="zero", **kwargs):
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
        is_decimated: bool, default True
            use a decimated or undecimated transform.
        axes: list of int, default None
            axes over which to compute the transform.
        padding_mode: str, default zero
            ways to extend the signal when computing the decomposition.
            See https://pywavelets.readthedocs.io/en/latest/ref/
            signal-extension-modes.html for more explanations.
        """
        # Inheritance
        super(PyWaveletTransformBase, self).__init__(
            nb_scale, verbose=verbose, dim=dim, **kwargs)

        # pywt Wavelet transform parameters
        self.is_decimated = is_decimated
        self.axes = axes
        if padding_mode not in pywt.Modes.modes:
            raise ValueError(
                "'{0}' is not a valid padding mode, should be one of "
                "{1}".format(padding_mode, pywt.Modes.modes))
        self.padding_mode = padding_mode

    def _init_transform(self, **kwargs):
        """ Define the transform.
        """
        if (self._pywt_func is None or self._pywt_name is None):
            raise ValueError("Transform not specified properly.")
        self.trf = self._pywt_func(self._pywt_name)

    def _analysis(self, data, **kwargs):
        """ Decompose a real signal using pywt.

        Parameters
        ----------
        data: nd-array
            a real array to be decomposed.

        Returns
        -------
        analysis_data: nd_array
            the decomposition coefficients.
        analysis_header: dict
            the decomposition associated information.
        """
        if self.is_decimated:
            coeffs = pywt.wavedecn(data, self.trf, mode=self.padding_mode,
                                   level=self.nb_scale, axes=self.axes)
        else:
            coeffs = pywt.swtn(data, self.trf, level=self.nb_scale,
                               axes=self.axes)
        analysis_data, analysis_header = self._organize_pysap(coeffs)
        self.nb_band_per_scale = [
            len(scale_info) for scale_info in analysis_header]

        return analysis_data, analysis_header

    def _synthesis(self, analysis_data, analysis_header):
        """ Reconstruct a real signal from the wavelet coefficients using pywt.

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
        coeffs = self._organize_pywt(analysis_data, analysis_header)
        if self.is_decimated:
            data = pywt.waverecn(coeffs, self.trf, mode=self.padding_mode,
                                 axes=self.axes)
        else:
            data = pywt.iswtn(coeffs, self.trf, axes=self.axes)
        return data

    def _organize_pysap(self, coeffs):
        """ Organize the coefficients from pywt for pysap.

        Parameters
        ----------
        coeffs: list of dict or ndarray
            the pywt input coefficents.

        Returns
        -------
        data: list of ndarray
            the organized coefficients.
        info: list
            the pywt transform information.
        """
        if not isinstance(coeffs, list):
            coeffs = [coeffs]
        elif len(coeffs) == 0:
            return None, None
        if self.is_decimated:
            coeffs[0] = {"a": coeffs[0]}
        data = []
        info = []
        for band_struct in coeffs:
            band_info = []
            for key, arr in band_struct.items():
                band_info.append((key, arr.shape))
                data.append(arr)
            info.append(band_info)

        return data, info

    def _organize_pywt(self, data, info):
        """ Organize the coefficients from pysap for pywt.

        Parameters
        ----------
        data: list of ndarray
            the organized coefficients.
        info: list
            the pywt transform information.

        Returns
        -------
        coeffs: list
            the pywt input coefficents.
        """
        coeffs = []
        offset = 0
        if self.is_decimated:
            coeffs.append(data[0])
            info = info[1:]
            offset += 1
        for band_struct in info:
            band_info = {}
            for cnt, (key, shape) in enumerate(band_struct):
                band_info[key] = data[cnt + offset]
            offset += len(band_struct)
            coeffs.append(band_info)

        return coeffs


def pywt_class_factory(func, name, destination_module_globals):
    """ Dynamically create a pywt transform.

    In order to make the class publicly accessible, we assign the result of
    the function to a variable dynamically using globals().

    Parameters
    ----------
    func: @function
        the wavelet transform function.
    name: str
        the wavelet name we want to instanciate.
    """
    # Define the transform class name
    class_name = name.replace(".", "")

    # Define the trsform class parameters
    class_parameters = {
        "__module__": destination_module_globals["__name__"],
        "_id":  destination_module_globals["__name__"] + "." + class_name,
        "_pywt_name": name,
        "_pywt_func": func
    }

    # Get the process instance associated to the function
    destination_module_globals[class_name] = (
        type(class_name, (PyWaveletTransformBase, ), class_parameters))


destination_module_globals = globals()
for family in pywt.families():
    if family in ("gaus", "mexh", "morl", "cgau", "shan", "fbsp", "cmor"):
        func = pywt.ContinuousWavelet
    else:
        func = pywt.Wavelet
    for name in pywt.wavelist(family):
        pywt_class_factory(func, name, destination_module_globals)


class ISAPWaveletTransformBase(WaveletTransformBase):
    """ Define the structure that will be used to store the ISAP results.
    """
    __family__ = "isap-2d"
    __isap_transform_id__ = None
    __isap_name__ = None
    __is_decimated__ = None
    __isap_nb_bands__ = None
    __isap_scale_shift__ = 0
    __mods__ = ["zero", "constant", "symmetric", "periodic"]

    def __init__(self, nb_scale, verbose=0, dim=2, padding_mode="zero",
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
        padding_mode: str, default zero
            ways to extend the signal when computing the decomposition.
        """
        # ISAP Wavelet transform parameters
        if hasattr(self, "__family__") and self.__family__ in ("isap-3d", ):
            dim = 3
        self.bands_lengths = None
        self.bands_shapes = None
        self.isap_transform_id = None
        self.flatten_fct = None
        self.unflatten_fct = None
        self.scales_lengths = None
        self.scales_padds = None
        if padding_mode not in self.__mods__:
            raise ValueError(
                "'{0}' is not a valid padding mode, should be one of "
                "{1}".format(padding_mode, self.__mods__))
        self.padding_mode = self.__mods__.index(padding_mode)

        # Inheritance
        super(ISAPWaveletTransformBase, self).__init__(
            nb_scale, verbose=verbose, dim=dim, use_wrapping=pysparse is None,
            **kwargs)

    def _init_transform(self, **kwargs):
        """ Define the transform.
        """
        if not self.use_wrapping:
            kwargs["type_of_multiresolution_transform"] = (
                self.__isap_transform_id__)
            kwargs["number_of_scales"] = self.nb_scale
            if self.data_dim == 2:
                kwargs["bord"] = self.padding_mode
                self.trf = pysparse.MRTransform(**kwargs)
            elif self.data_dim == 3:
                self.trf = pysparse.MRTransform3D(**kwargs)
            else:
                raise NameError("Please define a correct dimension for data.")
        else:
            if self.data_dim == 2:
                self.trf = None
            else:
                raise NameError("For {0}D, only the bindings work for "
                                "now.".format(self.data_dim))

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
        # Update ISAP parameters
        kwargs["type_of_multiresolution_transform"] = self.isap_transform_id
        kwargs["number_of_scales"] = self.nb_scale

        # Use subprocess to execute binaries
        if self.use_wrapping:
            kwargs["verbose"] = self.verbose > 0
            with pysap.TempDir(isap=True) as tmpdir:
                in_image = os.path.join(tmpdir, "in.fits")
                out_mr_file = os.path.join(tmpdir, "cube.mr")
                pysap.io.save(data, in_image)
                pysap.extensions.mr_transform(in_image, out_mr_file, **kwargs)
                image = pysap.io.load(out_mr_file)
                analysis_data = image.data
                analysis_header = image.metadata

            # Reorganize the generated coefficents
            self._analysis_shape = analysis_data.shape
            analysis_buffer = self.flatten_fct(analysis_data, self)
            self._analysis_buffer_shape = analysis_buffer.shape
            if not isinstance(self.nb_band_per_scale, list):
                self.nb_band_per_scale = (
                    self.nb_band_per_scale.squeeze().tolist())
            analysis_data = []
            for scale, nb_bands in enumerate(self.nb_band_per_scale):
                for band in range(nb_bands):
                    analysis_data.append(self._get_linear_band(
                        scale, band, analysis_buffer))

        # Use Python bindings
        else:
            analysis_data, self.nb_band_per_scale = self.trf.transform(
                data.astype(numpy.double), save=False)
            analysis_header = None

        return analysis_data, analysis_header

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
        # Use subprocess to execute binaries
        if self.use_wrapping:

            cube = pysap.Image(data=analysis_data[0], metadata=analysis_header)
            with pysap.TempDir(isap=True) as tmpdir:
                in_mr_file = os.path.join(tmpdir, "cube.mr")
                out_image = os.path.join(tmpdir, "out.fits")
                pysap.io.save(cube, in_mr_file)
                pysap.extensions.mr_recons(
                    in_mr_file, out_image, verbose=(self.verbose > 0))
                data = pysap.io.load(out_image).data

        # Use Python bindings
        else:
            data = self.trf.reconstruct(analysis_data)

        return data

    def _set_transformation_parameters(self):
        """ Declare transformation parameters.
        """
        # Check transformation has been defined
        if (self.__isap_transform_id__ is None or self.__isap_name__ is None
                or self.__is_decimated__ is None
                or self.__isap_nb_bands__ is None):

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
            (iso_shape * iso_shape)
            * numpy.ones((nb_scale, nb_band), dtype=int))
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
            bands_lengths[i] = scale / 2**(i + scale_shift)
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
            (self._iso_shape * self._iso_shape)
            * numpy.ones((nb_scale,  _map[self._iso_shape]), dtype=int))
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


class BiOrthogonalTransform3D(ISAPWaveletTransformBase):
    """ Mallat's 3D wavelet transform (7/9 biorthogonal filters)
    """
    __family__ = "isap-3d"
    __isap_transform_id__ = 1
    __isap_name__ = "3D Wavelet transform via lifting scheme"
    __is_decimated__ = True
    __isap_nb_bands__ = 7


class Wavelet3DTransformViaLiftingScheme(ISAPWaveletTransformBase):
    """ Wavelet transform via lifting scheme.
    """
    __family__ = "isap-3d"
    __isap_transform_id__ = 2
    __isap_name__ = "Wavelet transform via lifting scheme"
    __is_decimated__ = True
    __isap_nb_bands__ = 7


class ATrou3D(ISAPWaveletTransformBase):
    """ Wavelet transform with the A trou algorithm.
    """
    __family__ = "isap-3d"
    __isap_transform_id__ = 3
    __isap_name__ = "3D Wavelet A Trou"
    __is_decimated__ = False
    __isap_nb_bands__ = 1
