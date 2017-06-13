import unittest
import os.path as osp
import numpy as np

from pisap.numerics.noise import soft_thresholding, hard_thresholding


# global cst
SHAPE = (10, 10)
LEVEL = 0.5
REF = np.ones(SHAPE)
IMGS = [2.0*LEVEL*REF, # after the level
        1.0*LEVEL*REF, # on the level
        0.5*LEVEL*REF, # before the level
       ]
REF_CPLX = np.sqrt(2)/2 + 1.j*np.sqrt(2)/2 * np.ones(SHAPE)
IMGS_CPLX = [2.0*LEVEL*REF_CPLX, # modulus after the level
             1.0*LEVEL*REF_CPLX, # modulus on the level
             0.5*LEVEL*REF_CPLX, # modulus before the level
            ] 

class TestDenoise(unittest.TestCase):

    def test_soft_thresholding(self):
        # real case
        # after the level
        res = soft_thresholding(IMGS[0], LEVEL)
        np.testing.assert_allclose(res, IMGS[0] - LEVEL*REF)
        # on the level
        res = soft_thresholding(IMGS[1], LEVEL)
        np.testing.assert_allclose(res, IMGS[1] - LEVEL*REF)
        # before the level
        res = soft_thresholding(IMGS[2], LEVEL)
        np.testing.assert_allclose(res, np.zeros(SHAPE))
        # complex case
        # after the level
        res = soft_thresholding(IMGS_CPLX[0], LEVEL)
        np.testing.assert_allclose(res, IMGS_CPLX[0] - LEVEL*REF_CPLX)
        # on the level
        res = soft_thresholding(IMGS_CPLX[1], LEVEL)
        np.testing.assert_allclose(res, IMGS_CPLX[1] - LEVEL*REF_CPLX)
        # before the level
        res = soft_thresholding(IMGS_CPLX[2], LEVEL)
        np.testing.assert_allclose(res, np.zeros(SHAPE))

    def test_hard_thresholding(self):
        # complex and real case
        # after the level
        res = hard_thresholding(IMGS[0], LEVEL)
        np.testing.assert_allclose(res, IMGS[0])
        # on the level
        res = hard_thresholding(IMGS[1], LEVEL)
        np.testing.assert_allclose(res, IMGS[1])
        # before the level
        res = hard_thresholding(IMGS[2], LEVEL)
        np.testing.assert_allclose(res, np.zeros(SHAPE))
