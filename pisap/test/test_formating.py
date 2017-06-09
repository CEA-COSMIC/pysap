import unittest
import os.path as osp

import pisap
from pisap.numerics.linears import linearWaveletTransformATrousAlgorithm # 0
from pisap.numerics.linears import pyramidalBsplineWaveletTransform # 1
from pisap.numerics.linears import haarWaveletTransform # 2
from pisap.numerics.linears import fastCurveletTransform # 3
from pisap.numerics.linears import FeauveauWaveletTransform # 4


# global cst
IMG = pisap.io.load(osp.join("data", "M31_128.fits")).data, # 128*128 px


class TestFormating(unittest.TestCase):

    def test_flatten_undecimated_n_bands(self):
        pass
