import unittest
import numpy as np
import tempfile
import os
from scipy.io import loadmat

import pisap
from pisap.numerics.linears import *
from pisap.base.utils import run_both_trf


# global cst
imfile = "Iref.mat"
img = loadmat(imfile)['I']
min_buf = img.min()
max_buf = img.max()
IMG = (img - min_buf) / (max_buf - min_buf)
NB_SCALES = [2, 4, 6]
EPS = 1.0e-10


class TestTrf(unittest.TestCase):

    def test_linearWaveletTransformATrousAlgorithm(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 1,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        linearWaveletTransformATrousAlgorithm,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_bsplineWaveletTransformATrousAlgorithm(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 2,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        bsplineWaveletTransformATrousAlgorithm,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_waveletTransformInFourierSpace(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 3,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        waveletTransformInFourierSpace,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_morphologicalMedianTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 4,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        morphologicalMedianTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_morphologicalMinmaxTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 5,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        morphologicalMinmaxTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_pyramidalLinearWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 6,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        pyramidalLinearWaveletTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_pyramidalBsplineWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 7,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        pyramidalBsplineWaveletTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_pyramidalWaveletTransformInFourierSpaceAlgo1(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 8,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        pyramidalWaveletTransformInFourierSpaceAlgo1,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    @unittest.skip("Meyer's wavelets (compact support in Fourier space) skipped " \
                     + "because isap backend do not work")
    def test_MeyerWaveletsCompactInFourierSpace(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 9,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        MeyerWaveletsCompactInFourierSpace,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_pyramidalMedianTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 10,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        pyramidalMedianTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_pyramidalLaplacian(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 11,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        pyramidalLaplacian,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_morphologicalPyramidalMinmaxTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 12,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        morphologicalPyramidalMinmaxTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_decompositionOnScalingFunction(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 13,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        decompositionOnScalingFunction,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_MallatWaveletTransform7_9Filters(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 14,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        MallatWaveletTransform7_9Filters,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_FeauveauWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 15,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        FeauveauWaveletTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_FeauveauWaveletTransformWithoutUndersampling(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 16,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        FeauveauWaveletTransformWithoutUndersampling,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_LineColumnWaveletTransform1D1D(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 17,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        LineColumnWaveletTransform1D1D,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_haarWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 18,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        haarWaveletTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_halfPyramidalTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 19,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        halfPyramidalTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_mixedHalfPyramidalWTAndMedianMethod(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 20,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        mixedHalfPyramidalWTAndMedianMethod,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)


    def test_undecimatedDiadicWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 21,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        undecimatedDiadicWaveletTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_mixedWTAndPMTMethod(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 22,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        mixedWTAndPMTMethod,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_undecimatedHaarTransformATrousAlgorithm(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 23,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        undecimatedHaarTransformATrousAlgorithm,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_undecimatedBiOrthogonalTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 24,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        undecimatedBiOrthogonalTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

    def test_nonOrthogonalUndecimatedTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 25,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            pisap_trf_buf, isap_trf_buf = run_both_trf(
                                        nonOrthogonalUndecimatedTransform,
                                        IMG, nb_scale, isap_kwargs)
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)

if __name__ == '__main__':
    unittest.main()
