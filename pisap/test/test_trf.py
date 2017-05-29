import unittest
import numpy as np
import tempfile
import os
from scipy.io import loadmat

import pisap
from pisap.numerics.linears import *
from pisap.base.utils import run_both


# global cst
imfile = "Iref.mat" # test data: a true brain scan
img = loadmat(imfile)['I']
min_buf = img.min()
max_buf = img.max()
IMG = (img - min_buf) / (max_buf - min_buf) # to avoid 10^-10 values
NB_SCALES = [2, 3, 4]
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
            res = run_both(linearWaveletTransformATrousAlgorithm,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_bsplineWaveletTransformATrousAlgorithm(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 2,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(bsplineWaveletTransformATrousAlgorithm,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_waveletTransformInFourierSpace(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 3,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(waveletTransformInFourierSpace,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_morphologicalMedianTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 4,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(morphologicalMedianTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_morphologicalMinmaxTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 5,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(morphologicalMinmaxTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_pyramidalLinearWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 6,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(pyramidalLinearWaveletTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_pyramidalBsplineWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 7,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(pyramidalBsplineWaveletTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_pyramidalWaveletTransformInFourierSpaceAlgo1(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 8,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(pyramidalWaveletTransformInFourierSpaceAlgo1,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    @unittest.skip("Meyer's wavelets (compact support in Fourier space) skipped " \
                     + "because ISAP backend produce NaN")
    def test_MeyerWaveletsCompactInFourierSpace(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 9,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(MeyerWaveletsCompactInFourierSpace,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_pyramidalMedianTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 10,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(pyramidalMedianTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_pyramidalLaplacian(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 11,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(pyramidalLaplacian,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_morphologicalPyramidalMinmaxTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 12,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(morphologicalPyramidalMinmaxTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_decompositionOnScalingFunction(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 13,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(decompositionOnScalingFunction,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_MallatWaveletTransform7_9Filters(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 14,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(MallatWaveletTransform7_9Filters,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_FeauveauWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 15,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(FeauveauWaveletTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_FeauveauWaveletTransformWithoutUndersampling(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 16,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(FeauveauWaveletTransformWithoutUndersampling,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_LineColumnWaveletTransform1D1D(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 17,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(LineColumnWaveletTransform1D1D,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_haarWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 18,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(haarWaveletTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_halfPyramidalTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 19,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(halfPyramidalTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_mixedHalfPyramidalWTAndMedianMethod(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 20,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(mixedHalfPyramidalWTAndMedianMethod,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_undecimatedDiadicWaveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 21,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(undecimatedDiadicWaveletTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_mixedWTAndPMTMethod(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 22,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(mixedWTAndPMTMethod,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_undecimatedHaarTransformATrousAlgorithm(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 23,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(undecimatedHaarTransformATrousAlgorithm,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_undecimatedBiOrthogonalTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 24,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(undecimatedBiOrthogonalTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_nonOrthogonalUndecimatedTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 25,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(nonOrthogonalUndecimatedTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    @unittest.skip("Isotropic and compact support wavelet in Fourier space skipped " \
                     + "because ISAP backend produce NaN")
    def test_isotropicAndCompactSupportWaveletInFourierSpace(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 26,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(isotropicAndCompactSupportWaveletInFourierSpace,
                                       IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_pyramidalWaveletTransformInFourierSpaceAlgo2(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 27,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(pyramidalWaveletTransformInFourierSpaceAlgo2,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    @unittest.skip("Fast Curvelet Transform skipped because not set yet")
    def test_fastCurveletTransform(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 28,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(fastCurveletTransform,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_waveletTransformViaLiftingScheme(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 29,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(waveletTransformViaLiftingScheme,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_onLine5_3AndOnColumn4_4(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 30,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(onLine5_3AndOnColumn4_4,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)

    def test_onLine4_4AndOnColumn5_3(self):
        for nb_scale in NB_SCALES:
            isap_kwargs = {
                'type_of_multiresolution_transform': 31,
                'number_of_scales': nb_scale,
                'type_of_filters': 1,
                'type_of_non_orthog_filters': 2,
                'use_l2_norm': False,
                }
            res = run_both(onLine4_4AndOnColumn5_3,
                               IMG, nb_scale, isap_kwargs)
            pisap_trf_buf, isap_trf_buf = res[0]
            np.testing.assert_allclose(pisap_trf_buf, isap_trf_buf)
            pisap_recs_img, isap_recs_img = res[1]
            np.testing.assert_allclose(pisap_recs_img, isap_recs_img)


if __name__ == '__main__':
    unittest.main()
