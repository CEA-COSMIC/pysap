# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import numpy
import warnings

# Package import
from pysap.plugins.mri.reconstruct.fourier import FFT2, NFFT2
from pysap.plugins.mri.reconstruct_3D.fourier import FFT3, NFFT3
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.reconstruct.utils import convert_locations_to_mask
from pysap.plugins.mri.reconstruct.utils import normalize_frequency_locations
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.reconstruct_3D.utils import convert_locations_to_mask_3D


class TestAdjointOperatorFourierTransform(unittest.TestCase):
    """ Test the adjoint operator of the NFFT both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 64
        self.max_iter = 10

    def test_normalize_frequency_locations(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed between [-0.5; 0.5[
        """
        for _ in range(10):
            samples = numpy.random.randn(128*128, 2)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertFalse((normalized_samples.all() < 0.5 and
                             normalized_samples.all() >= -0.5))
        print(" Test normalization function")

    def test_sampling_converters(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            print("Process test convert mask to samples test '{0}'...", i)
            Nx = numpy.random.randint(8, 512)
            Ny = numpy.random.randint(8, 512)
            mask = numpy.random.randint(2, size=(Nx, Ny))
            samples = convert_mask_to_locations(mask)
            recovered_mask = convert_locations_to_mask(samples,
                                                       (Nx, Ny))
            self.assertEqual(mask.all(), recovered_mask.all())
            mismatch = 0. + (numpy.mean(
                numpy.allclose(mask, recovered_mask)))
            print("      mismatch = ", mismatch)
        print(" Test convert mask to samples and it's adjoint passes")

    def test_sampling_converters_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            print("Process test convert mask to samples in 3D test '{0}'...",
                  i)
            Nx = numpy.random.randint(8, 512)
            Ny = numpy.random.randint(8, 512)
            Nz = numpy.random.randint(8, 512)
            mask = numpy.random.randint(2, size=(Nx, Ny, Nz))
            samples = convert_mask_to_locations_3D(mask)
            recovered_mask = convert_locations_to_mask_3D(samples,
                                                          (Nx, Ny, Nz))
            self.assertEqual(mask.all(), recovered_mask.all())
            mismatch = 0. + (numpy.mean(
                numpy.allclose(mask, recovered_mask)))
            print("      mismatch = ", mismatch)
        print(" Test convert mask to samples and it's adjoint passes in 3D")

    def test_FFT2(self):
        """Test the adjoint operator for the 2D Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process FFT2 test '{0}'...", i)
            fourier_op_dir = FFT2(samples=_samples, shape=(self.N, self.N))
            fourier_op_adj = FFT2(samples=_samples, shape=(self.N, self.N))
            Img = numpy.random.randn(self.N, self.N)
            f = numpy.random.randn(self.N, self.N)
            f_p = fourier_op_dir.op(Img) / (self.N ** 2)
            I_p = fourier_op_adj.adj_op(f)
            x_d = numpy.dot(Img.flatten(), numpy.conj(I_p).flatten())
            x_ad = numpy.dot(f_p.flatten(), numpy.conj(f).flatten())
            self.assertTrue(numpy.isclose(x_d, x_ad, rtol=1e-3))
            mismatch = (1. - numpy.mean(
                numpy.isclose(x_d, x_ad,
                              rtol=1e-3)))
            print("      mismatch = ", mismatch)
        print(" FFT2 adjoint test passes")

    def test_NFFT2(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        # warnings.warn('No test will be made for the NFFT package')
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process NFFT2 test '{0}'...", i)
            fourier_op_dir = NFFT2(samples=_samples, shape=(self.N, self.N))
            fourier_op_adj = NFFT2(samples=_samples, shape=(self.N, self.N))
            Img = numpy.random.randn(self.N, self.N) + \
                1j * numpy.random.randn(self.N, self.N)
            f = numpy.random.randn(_samples.shape[0], 1) + \
                1j * numpy.random.randn(_samples.shape[0], 1)
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = numpy.dot(Img.flatten(), numpy.conj(I_p).flatten())
            x_ad = numpy.dot(f_p.flatten(), numpy.conj(f).flatten())
            self.assertTrue(numpy.isclose(x_d, x_ad, rtol=1e-3))
            mismatch = (1. - numpy.mean(
                numpy.isclose(x_d, x_ad,
                              rtol=1e-3)))
            print("      mismatch = ", mismatch)
        print(" NFFT2 adjoint test passes")

    def test_FFT3(self):
        """Test the adjoint operator for the 3D Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N, self.N))
            _samples = convert_mask_to_locations_3D(_mask)
            print("Process FFT3 test '{0}'...", i)
            fourier_op_dir = FFT3(samples=_samples,
                                  shape=(self.N, self.N, self.N))
            fourier_op_adj = FFT3(samples=_samples,
                                  shape=(self.N, self.N, self.N))
            Img = numpy.random.randn(self.N, self.N, self.N)
            f = numpy.random.randn(self.N, self.N, self.N)
            f_p = fourier_op_dir.op(Img) / (self.N ** 3)
            I_p = fourier_op_adj.adj_op(f)
            x_d = numpy.dot(Img.flatten(), numpy.conj(I_p).flatten())
            x_ad = numpy.dot(f_p.flatten(), numpy.conj(f).flatten())
            self.assertTrue(numpy.isclose(x_d, x_ad, rtol=1e-3))
            mismatch = (1. - numpy.mean(
                numpy.isclose(x_d, x_ad,
                              rtol=1e-3)))
            print("      mismatch = ", mismatch)
        print(" FFT3 adjoint test passes")

    def test_NFFT3(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        # warnings.warn('No tests will be done on the NFFT operator')
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N, self.N))
            _samples = convert_mask_to_locations_3D(_mask)
            print("Process NFFT3 test '{0}'...", i)
            fourier_op_dir = NFFT3(samples=_samples,
                                   shape=(self.N, self.N, self.N))
            fourier_op_adj = NFFT3(samples=_samples,
                                   shape=(self.N, self.N, self.N))
            Img = numpy.random.randn(self.N, self.N, self.N) + \
                1j * numpy.random.randn(self.N, self.N, self.N)
            f = numpy.random.randn(_samples.shape[0], 1) + \
                1j * numpy.random.randn(_samples.shape[0], 1)
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = numpy.dot(Img.flatten(), numpy.conj(I_p).flatten())
            x_ad = numpy.dot(f_p.flatten(), numpy.conj(f).flatten())
            self.assertTrue(numpy.isclose(x_d, x_ad, rtol=1e-3))
            mismatch = (1. - numpy.mean(
                numpy.isclose(x_d, x_ad,
                              rtol=1e-3)))
            print("      mismatch = ", mismatch)
        print(" NFFT3 adjoint test passes")


if __name__ == "__main__":
    unittest.main()
