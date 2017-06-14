import unittest
import os.path as osp
import numpy as np
import scipy.fftpack as pfft

import pisap
from pisap.numerics.linears import haarWaveletTransform, \
                                   pyramidalWaveletTransformInFourierSpaceAlgo2


# global cst
IMGS = [pisap.io.load(osp.join("data", "M31_128.fits")).data, # 128*128 px
        pisap.io.load(osp.join("data", "M31_256_128.fits")).data, # 256*128 px
       ]


class TestDictionary(unittest.TestCase):

    def test_img_shape(self):
        linear = haarWaveletTransform(**{'maxscale': 4})
        self.assertRaisesRegexp(ValueError,
                                "in 'DictionaryBase': 'metadata' " \
                                 + "passed for init is not valid.",
                                linear.op, IMGS[1]) # 256*128 px

    def test_too_much_decimated(self):
        # ok maxscale
        linear = haarWaveletTransform(**{'maxscale': 7})
        try:
            linear.op(IMGS[0])
        except ValueError as e:
            self.fail("unexpected 'ValueError: {0}' occur in linear.op(IMGS[0])".format(e))
        # wrong maxscale
        linear = haarWaveletTransform(**{'maxscale': 8})
        self.assertRaisesRegexp(ValueError,
                                "in 'DictionaryBase': 'metadata' " \
                                 + "passed for init is not valid.",
                                linear.op, IMGS[0])
        # wrong maxscale
        linear = haarWaveletTransform(**{'maxscale': 9})
        self.assertRaisesRegexp(ValueError,
                                "in 'DictionaryBase': 'metadata' " \
                                 + "passed for init is not valid.",
                                linear.op, IMGS[0])
        # wrong maxscale
        linear = haarWaveletTransform(**{'maxscale': 10})
        self.assertRaisesRegexp(ValueError,
                                "in 'DictionaryBase': 'metadata' " \
                                 + "passed for init is not valid.",
                                linear.op, IMGS[0])

    def test_same_metadata(self):
        # 1 ref, 1 positif, 2 negatifs
        linear1 = haarWaveletTransform(**{'maxscale': 6})
        trf1 = linear1.op(IMGS[0])
        linear2 = haarWaveletTransform(**{'maxscale': 6})
        trf2 = linear2.op(IMGS[0])
        linear3 = haarWaveletTransform(**{'maxscale': 7})
        trf3 = linear3.op(IMGS[0])
        linear4 = pyramidalWaveletTransformInFourierSpaceAlgo2(**{'maxscale': 6})
        trf4 = linear4.op(IMGS[0])
        # tests
        self.assertTrue(trf1.check_same_metadata(trf1))
        self.assertTrue(trf1.check_same_metadata(trf2))
        self.assertFalse(trf1.check_same_metadata(trf3))
        self.assertFalse(trf1.check_same_metadata(trf4))

    def test_invalid_data(self):
        # valid case
        linear = haarWaveletTransform(**{'maxscale': 6})
        trf = linear.op(IMGS[0])
        self.assertTrue(trf.data_is_valid())
        # NaN case
        trf = trf * np.nan
        self.assertFalse(trf.data_is_valid())
        # None case
        trf._data = None
        self.assertFalse(trf.data_is_valid())

    def test_greater_or_equal(self):
        # 1 ref, 1 complex, 1 2-times greater
        linear = haarWaveletTransform(**{'maxscale': 6})
        trf = linear.op(IMGS[0])
        trf_abs = trf.absolute
        trf_complex = linear.op(pfft.ifft2(pfft.fft2(IMGS[0])))
        # tests
        res = (trf >= trf)
        self.assertTrue(np.all(res._data))

        self.assertRaisesRegexp(ValueError,
                                "Cannot compare '>=' complex.",
                                trf.__ge__, trf_complex)

        res = (trf >= trf_abs)
        self.assertFalse(np.all(res._data))

        res = (trf_abs >= trf)
        self.assertTrue(np.all(res._data))

        res = (trf >= 10.0)
        self.assertFalse(np.all(res._data))

        self.assertRaisesRegexp(ValueError,
                                "Cannot compare '>=' complex.",
                                trf_complex.__ge__, 10.0 + 10.0*1.j)

        self.assertRaisesRegexp(ValueError,
                                "Cannot compare '>=' complex.",
                                trf.__ge__, 10.0 + 10.0*1.j)

        self.assertRaisesRegexp(ValueError,
                                "type of 'other' to compare >= not understood",
                                trf.__ge__, "banana")

    def test_add(self):
        linear = haarWaveletTransform(**{'maxscale': 6})
        trf = linear.op(IMGS[0])
        np.testing.assert_allclose((trf+trf)._data, (2*trf)._data)

    def test_sub(self):
        linear = haarWaveletTransform(**{'maxscale': 6})
        trf = linear.op(IMGS[0])
        self.assertTrue(np.all((trf-trf)._data == 0.0))

    def test_mul(self):
        nb_scale = 4
        linear = haarWaveletTransform(**{'maxscale': nb_scale})
        trf = linear.op(IMGS[0])

        # case real scalar
        coef = 2.0
        np.testing.assert_allclose((coef * trf)._data, coef * trf._data)

        # case cplx scalar
        coef_complex = 1.0 + 1.j
        np.testing.assert_allclose((coef_complex * trf)._data,
                                      coef_complex * trf._data)

        # case list of real
        coefs = list(np.linspace(-5, 5, nb_scale))
        new_trf = trf * coefs
        for ks in range(new_trf.nb_scale):
            np.testing.assert_allclose(new_trf.get_scale(ks),
                                          coefs[ks] * trf.get_scale(ks))

        # case list of cplx
        coefs_complex = list(1.j*np.linspace(-5, 5, nb_scale))
        new_trf = trf * coefs_complex
        for ks in range(new_trf.nb_scale):
            np.testing.assert_allclose(new_trf.get_scale(ks),
                                          coefs_complex[ks] * trf.get_scale(ks))

        # case other DictinaryBase
        np.testing.assert_allclose(np.sqrt((trf.absolute * trf.absolute)._data),
                                      trf.absolute._data)

    def test_div(self):
        nb_scale = 4
        linear = haarWaveletTransform(**{'maxscale': nb_scale})
        trf = linear.op(IMGS[0])

        # case real scalar
        coef = 2.0
        np.testing.assert_allclose((trf / coef)._data, trf._data / coef)

        # case cplx scalar
        coef_complex = 1.0 + 1.j
        np.testing.assert_allclose((trf / coef_complex)._data,
                                       trf._data / coef_complex)

        # case list of real
        coefs = list(np.linspace(-5, 5, nb_scale))
        new_trf = trf / coefs
        for ks in range(new_trf.nb_scale):
            np.testing.assert_allclose(new_trf.get_scale(ks),
                                          trf.get_scale(ks) / coefs[ks])

        ## case list of cplx
        coefs_complex = list(1.j*np.linspace(-5, 5, nb_scale))
        new_trf = trf / coefs_complex
        for ks in range(new_trf.nb_scale):
            np.testing.assert_allclose(new_trf.get_scale(ks),
                                          trf.get_scale(ks) / coefs_complex[ks])

        # case other DictinaryBase
        trf._data = trf._data + 10.0 # to avoid 0
        np.testing.assert_allclose((trf / trf)._data,
                                      np.ones_like(trf._data))
