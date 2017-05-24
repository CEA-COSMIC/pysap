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
import numpy as np
from pisap.base.dictionary import Dictionary
from pisap.base.utils import set_bands_shapes


class linearWaveletTransformATrousAlgorithm(Dictionary):
    """ linear wavelet transform: a trous algorithm.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "linear wavelet transform: a trous algorithm"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 1
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class bsplineWaveletTransformATrousAlgorithm(Dictionary):
    """ bspline wavelet transform: a trous algorithm.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "linear wavelet transform: a trous algorithm"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 2
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class waveletTransformInFourierSpace(Dictionary):
    """ wavelet transform in Fourier space.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "wavelet transform in Fourier space"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 3
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class morphologicalMedianTransform(Dictionary):
    """ morphological median transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "morphological median transform"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 4
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class morphologicalMinmaxTransform(Dictionary):
    """ morphological minmax transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "morphological minmax transform"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 5
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class pyramidalLinearWaveletTransform(Dictionary):
    """ pyramidal linear wavelet transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "pyramidal linear wavelet transform"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 6
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class pyramidalBsplineWaveletTransform(Dictionary):
    """ pyramidal bspline wavelet transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "pyramidal bspline wavelet transform"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 7
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class pyramidalWaveletTransformInFourierSpaceAlgo1(Dictionary):
    """ pyramidal wavelet transform in Fourier space: algo 1
        (diff. between two resolutions).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "pyramidal wavelet transform in Fourier space: algo 1" \
                + "(diff. between two resolutions)"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 8
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class MeyerWaveletsCompactInFourierSpace(Dictionary):
    """ Meyers wavelets (compact support in Fourier space).
    """
    def _trf_id(self):
        raise NotImplementedError("WIP: produce NaN")
        nb_scale = self.metadata['nb_scale']
        # name
        name = "Meyers wavelets (compact support in Fourier space)"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 9
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class pyramidalMedianTransform(Dictionary):
    """ pyramidal median transform (PMT).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "pyramidal median transform (PMT)"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 10
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class pyramidalLaplacian(Dictionary):
    """ pyramidal laplacian.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "pyramidal laplacian"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 11
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class morphologicalPyramidalMinmaxTransform(Dictionary):
    """ morphological pyramidal minmax transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "morphological pyramidal minmax transform"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 12
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class decompositionOnScalingFunction(Dictionary):
    """ decomposition on scaling function.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "decomposition on scaling function"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 13
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class MallatWaveletTransform7_9Filters(Dictionary):
    """ Mallat's wavelet transform (7/9 filters).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "Mallat's wavelet transform (7/9 filters)"
        # bands_names
        bands_names = 'v', 'd', 'h'
        # nb_band_per_scale
        nb_band_per_scale = np.array([3] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 3))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**(i+1)
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 14
        # type of from_cube
        id_formating = 2
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class FeauveauWaveletTransform(Dictionary):
    """ Feauveau's wavelet transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "Feauveau's wavelet transform"
        # bands_names
        bands_names = 'd1', 'd2'
        # nb_band_per_scale
        nb_band_per_scale = np.array([2] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 2))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**(i+1)
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        bands_lengths[:, 1] *= 2
        # bands_shapes
        ratios = np.ones_like(bands_lengths, dtype=float)
        ratios[:, 1] *= 2.0
        bands_shapes = set_bands_shapes(bands_lengths, ratio=ratios)
        # idx tdf
        id_trf = 15
        # type of from_cube
        id_formating = 4
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class FeauveauWaveletTransformWithoutUndersampling(Dictionary):
    """ Feauveau's wavelet transform without undersampling.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "Feauveau's wavelet transform without undersampling"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 16
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class LineColumnWaveletTransform1D1D(Dictionary):
    """ Line Column Wavelet Transform (1D+1D).
    """
    def _trf_id(self):
        nb_scale = 6
        # name
        name = "Line Column Wavelet Transform (1D+1D)"
        # bands_names
        bands_names = ['d%d'%i for i in range(6)]
        # nb_band_per_scale
        nb_band_per_scale = np.array([6] * nb_scale)
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 6))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 17
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class haarWaveletTransform(Dictionary):
    """ Haar's wavelet transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "linear wavelet transform: a trous algorithm"
        # bands_names
        bands_names = 'v', 'd', 'h'
        # nb_band_per_scale
        nb_band_per_scale = np.array([3] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 3))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**(i+1)
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 18
        # type of from_cube
        id_formating = 2
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class halfPyramidalTransform(Dictionary):
    """ half-pyramidal transform.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "half-pyramidal transform"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 19
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class mixedHalfPyramidalWTAndMedianMethod(Dictionary):
    """ mixed Half-pyramidal WT and Median method (WT-HPMT).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "mixed Half-pyramidal WT and Median method (WT-HPMT)"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 20
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class undecimatedDiadicWaveletTransform(Dictionary):
    """ undecimated diadic wavelet transform (two bands per scale).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "undecimated diadic wavelet transform (two bands per scale)"
        # bands_names
        bands_names = 'd1', 'd2'
        # nb_band_per_scale
        nb_band_per_scale = np.array([2] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 2))
        bands_lengths[-1, 1:] = 0
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 21
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class mixedWTAndPMTMethod(Dictionary):
    """ mixed WT and PMT method (WT-PMT).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "mixed WT and PMT method (WT-PMT)"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = nx * np.ones((nb_scale, 1))
        bands_lengths[-1,1:] = 0
        for i, scale in enumerate(bands_lengths):
            scale /= 2**i
        bands_lengths[-1,:] *= 2
        bands_lengths = (bands_lengths**2).astype(int)
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 22
        # type of from_cube
        id_formating = 1
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class undecimatedHaarTransformATrousAlgorithm(Dictionary):
    """ undecimated Haar transform: a trous algorithm (one band per scale).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "undecimated Haar transform: a trous algorithm (one band per scale)"
        # bands_names
        bands_names = 'a'
        # nb_band_per_scale
        nb_band_per_scale = np.array([1] * (nb_scale-1) + [1])
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 1))
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 23
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class undecimatedBiOrthogonalTransform(Dictionary):
    """ undecimated (bi-) orthogonal transform (three bands per scale.
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "undecimated (bi-) orthogonal transform (three bands per scale"
        # bands_names
        bands_names = 'v', 'd', 'h'
        # nb_band_per_scale
        nb_band_per_scale = np.array([3] * nb_scale)
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 3))
        bands_lengths[-1, 1] = 0.
        bands_lengths[-1, 2] = 0.
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 24
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating


class nonOrthogonalUndecimatedTransform(Dictionary):
    """ non orthogonal undecimated transform (three bands per scale).
    """
    def _trf_id(self):
        nb_scale = self.metadata['nb_scale']
        # name
        name = "non orthogonal undecimated transform (three bands per scale)"
        # bands_names
        bands_names = 'v', 'd', 'h'
        # nb_band_per_scale
        nb_band_per_scale = np.array([3] * nb_scale)
        # bands_lengths
        nx, _ = self.data.shape
        bands_lengths = (nx * nx) * np.ones((nb_scale, 3))
        bands_lengths[-1, 1] = 0.
        bands_lengths[-1, 2] = 0.
        # bands_shapes
        bands_shapes = set_bands_shapes(bands_lengths)
        # idx tdf
        id_trf = 25
        # type of from_cube
        id_formating = 0
        return name, bands_names, nb_band_per_scale, bands_lengths, \
               bands_shapes, id_trf, id_formating
