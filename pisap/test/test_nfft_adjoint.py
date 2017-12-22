##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import numpy

# Package import
from pisap.plugins.mri.reconstruct.fourier import NFFT2
from pisap.plugins.mri.reconstruct.utils import convert_mask_to_locations


class TestAdjointOperatorFourierTransform(unittest.TestCase):
    """ Test the adjoint operator of the NFFT both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 256
        self.max_iter = 10

    def test_NFFT2(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process NFFT2 test '{0}'...", i)
            fourier_op_dir = NFFT2(samples=_samples, shape=(self.N, self.N))
            fourier_op_adj = NFFT2(samples=_samples, shape=(self.N, self.N))
            Img = numpy.random.randn(self.N, self.N)
            f = numpy.random.randn(len(_samples))
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = numpy.dot(I_p.flatten(), numpy.conj(Img.flatten()))
            x_ad = numpy.dot(f, numpy.conj(f_p))/len(_samples)
            mismatch = (1. - numpy.mean(
                numpy.isclose(x_d, x_ad,
                              rtol=1e-3)))
            print("      mismatch = ", mismatch)
        print(" NFFT2 adjoint test passes")


if __name__ == "__main__":
    unittest.main()
