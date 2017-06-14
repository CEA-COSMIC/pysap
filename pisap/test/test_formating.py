import unittest
import os.path as osp
import numpy as np

import pisap
from pisap.base.utils import isap_transform
from pisap.numerics.linears import linearWaveletTransformATrousAlgorithm, \
                                   pyramidalBsplineWaveletTransform, \
                                   haarWaveletTransform, \
                                   fastCurveletTransform, \
                                   FeauveauWaveletTransform
from pisap.base.formating import flatten_undecimated_n_bands, \
                                 flatten_decimated_1_bands, \
                                 flatten_decimated_3_bands, \
                                 flatten_vector, \
                                 flatten_decimated_feauveau, \
                                 inflated_undecimated_n_bands, \
                                 inflated_decimated_1_bands, \
                                 inflated_decimated_3_bands, \
                                 inflated_vector, \
                                 inflated_decimated_feauveau


# global cst
IMG = pisap.io.load(osp.join("data", "M31_128.fits")).data # 128*128 px
NB_SCALE = 3

class TestFormating(unittest.TestCase):

    def test_undecimated_n_bands(self):
        # set variables
        id_trf = 1
        isap_kwargs = {'type_of_multiresolution_transform': id_trf,
                       'number_of_scales': NB_SCALE,
                      }
        linear_op = linearWaveletTransformATrousAlgorithm(maxscale=NB_SCALE)
        trf = linear_op.op(IMG)
        # test
        ref_cube, header = isap_transform(IMG, **isap_kwargs)
        flatten_cube = flatten_undecimated_n_bands(ref_cube, trf)
        trf._data = flatten_cube
        inflated_cube = inflated_undecimated_n_bands(trf)
        np.testing.assert_allclose(inflated_cube, ref_cube)

    def test_decimated_1_bands(self):
        # set variables
        id_trf = 7
        isap_kwargs = {'type_of_multiresolution_transform': id_trf,
                       'number_of_scales': NB_SCALE,
                      }
        linear_op = pyramidalBsplineWaveletTransform(maxscale=NB_SCALE)
        trf = linear_op.op(IMG)
        # test
        ref_cube, header = isap_transform(IMG, **isap_kwargs)
        flatten_cube = flatten_decimated_1_bands(ref_cube, trf)
        trf._data = flatten_cube
        inflated_cube = inflated_decimated_1_bands(trf)
        np.testing.assert_allclose(inflated_cube, ref_cube)

    def test_decimated_3_bands(self):
        # set variables
        id_trf = 18
        isap_kwargs = {'type_of_multiresolution_transform': id_trf,
                       'number_of_scales': NB_SCALE,
                      }
        linear_op = haarWaveletTransform(maxscale=NB_SCALE)
        trf = linear_op.op(IMG)
        # test
        ref_cube, header = isap_transform(IMG, **isap_kwargs)
        flatten_cube = flatten_decimated_3_bands(ref_cube, trf)
        trf._data = flatten_cube
        inflated_cube = inflated_decimated_3_bands(trf)
        np.testing.assert_allclose(inflated_cube, ref_cube)

    def test_vector(self):
        # set variables
        id_trf = 28
        isap_kwargs = {'type_of_multiresolution_transform': id_trf,
                       'number_of_scales': NB_SCALE,
                      }
        linear_op = fastCurveletTransform(maxscale=NB_SCALE)
        trf = linear_op.op(IMG)
        # test
        ref_cube, header = isap_transform(IMG, **isap_kwargs)
        flatten_cube = flatten_vector(ref_cube, trf)
        trf._data = flatten_cube
        inflated_cube = inflated_vector(trf)
        np.testing.assert_allclose(inflated_cube, ref_cube)

    def test_decimated_feauveau(self):
        # set variables
        id_trf = 15
        isap_kwargs = {'type_of_multiresolution_transform': id_trf,
                       'number_of_scales': NB_SCALE,
                      }
        linear_op = FeauveauWaveletTransform(maxscale=NB_SCALE)
        trf = linear_op.op(IMG)
        # test
        ref_cube, header = isap_transform(IMG, **isap_kwargs)
        flatten_cube = flatten_decimated_feauveau(ref_cube, trf)
        trf._data = flatten_cube
        inflated_cube = inflated_decimated_feauveau(trf)
        np.testing.assert_allclose(inflated_cube, ref_cube)
