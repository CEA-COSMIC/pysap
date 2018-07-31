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
from scipy.fftpack import fftshift


# Package import
import pysap
from pysap.numerics.fourier import FFT2
from pysap.numerics.fourier import NFFT2
from pysap.numerics.linear import Wavelet2
from pysap.numerics.proximity import Threshold
from pysap.numerics.gradient import GradAnalysis2
from pysap.numerics.gradient import GradSynthesis2
from pysap.numerics.reconstruct import sparse_rec_fista
from pysap.numerics.reconstruct import sparse_rec_condatvu
from pysap.numerics.utils import convert_mask_to_locations

import pysap.extensions.transform
import warnings
from pysap.data import get_sample_data


class TestOptimizer(unittest.TestCase):
    """ Test the FISTA's gradient descent.
    """
    def setUp(self):
        """ Get the data from the server.
        """
        self.images = [get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [im.data.shape for im in self.images]))
        self.mask = get_sample_data("mri-mask").data
        self.names = ["MallatWaveletTransform79Filters"]
        print("[info] Found {0} transformations.".format(len(self.names)))
        self.nb_scales = [4]
        self.nb_iter = 100

    def test_reconstruction_fista_fft2(self):
        """ Test all the registered transformations.
        """
        print("Process test FFT2 FISTA::")
        for image in self.images:
            fourier = FFT2(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                           shape=image.shape)
            data = fourier.op(image.data)
            fourier_op = FFT2(convert_mask_to_locations(
                                            fftshift(self.mask)),
                              shape=image.shape)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for name in self.names:
                    print("    Transform: {0}".format(name))
                    linear_op = Wavelet2(
                        wavelet_name=name,
                        nb_scale=4)
                    gradient_op = GradSynthesis2(
                        data=data,
                        fourier_op=fourier_op,
                        linear_op=linear_op)
                    prox_op = Threshold(None)
                    x_final, transform, _, _ = sparse_rec_fista(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_op=prox_op,
                        cost_op=None,
                        mu=0,
                        lambda_init=1.0,
                        max_nb_of_iter=self.nb_iter,
                        atol=1e-4,
                        verbose=0)
                    self.assertTrue(numpy.isclose(x_final,
                                                  fourier.adj_op(data),
                                                  rtol=1e-3).all())
                    mismatch = (1. - numpy.mean(
                        numpy.isclose(x_final, fourier.adj_op(data),
                                      rtol=1e-3)))
                    print("      mismatch = ", mismatch)

    def test_reconstruction_condat_vu_fft2(self):
        """ Test all the registered transformations.
        """
        print("Process test FFT2 Condat Vu algorithm::")
        for image in self.images:
            fourier = FFT2(samples=convert_mask_to_locations(
                                fftshift(self.mask)), shape=image.shape)
            data = fourier.op(image.data)
            fourier_op = FFT2(samples=convert_mask_to_locations(
                                fftshift(self.mask)), shape=image.shape)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for name in self.names:
                    print("    Transform: {0}".format(name))
                    linear_op = Wavelet2(
                        wavelet_name=name,
                        nb_scale=4)
                    gradient_op = GradAnalysis2(
                        data=data,
                        fourier_op=fourier_op)
                    prox_dual_op = Threshold(None)
                    x_final, transform, _, _ = sparse_rec_condatvu(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_dual_op=prox_dual_op,
                        cost_op=None,
                        std_est=0.0,
                        std_est_method="dual",
                        std_thr=0,
                        mu=0,
                        tau=None,
                        sigma=None,
                        relaxation_factor=1.0,
                        nb_of_reweights=0,
                        max_nb_of_iter=self.nb_iter,
                        add_positivity=False,
                        atol=1e-4,
                        verbose=0)
                    self.assertTrue(numpy.isclose(x_final,
                                                  fourier.adj_op(data),
                                                  rtol=1e-3).all())
                    mismatch = (1. - numpy.mean(
                        numpy.isclose(x_final, fourier.adj_op(data),
                                      rtol=1e-3)))
                    print("      mismatch = ", mismatch)
                    return

    def test_reconstruction_fista_nfft2(self):
        """ Test all the registered transformations.
        """
        warnings.warn('No test will be mage on the NFFT package')
        # print('Process test NFFT2 FISTA')
        # for image in self.images:
        #     fourier = NFFT2(samples=convert_mask_to_locations(self.mask),
        #                     shape=image.shape)
        #     data = fourier.op(image.data)
        #     I_0_fourier = FFT2(samples=convert_mask_to_locations(
        #                        fftshift(self.mask)),
        #                     shape=image.shape)
        #     I_0 = I_0_fourier.adj_op(I_0_fourier.op(image.data))
        #     fourier_op = NFFT2(convert_mask_to_locations(self.mask),
        #                        shape=image.shape)
        #     print("Process test with image '{0}'...".format(
        #         image.metadata["path"]))
        #     for nb_scale in self.nb_scales:
        #         print("- Number of scales: {0}".format(nb_scale))
        #         for name in self.names:
        #             print("    Transform: {0}".format(name))
        #             linear_op = Wavelet2(
        #                 wavelet_name=name,
        #                 nb_scale=4)
        #             gradient_op = GradSynthesis2(
        #                 data=data,
        #                 fourier_op=fourier_op,
        #                 linear_op=linear_op)
        #             prox_op = Threshold(None)
        #             x_final, transform, _, _ = sparse_rec_fista(
        #                 gradient_op=gradient_op,
        #                 linear_op=linear_op,
        #                 prox_op=prox_op,
        #                 cost_op=None,
        #                 mu=0,
        #                 lambda_init=1.0,
        #                 max_nb_of_iter=self.nb_iter,
        #                 atol=1e-4,
        #                 verbose=0)
        #             mismatch = (1. - numpy.mean(
        #                 numpy.isclose(x_final, I_0,
        #                               rtol=1e-3)))
        #             print("      mismatch = ", mismatch)

    def test_reconstruction_condat_vu_nfft2(self):
        """ Test all the registered transformations.
        """
        warnings.warn('No test will be mage on the NFFT package')
        # for image in self.images:
        #     fourier = NFFT2(samples=convert_mask_to_locations(
        #                         fftshift(self.mask)), shape=image.shape)
        #     data = fourier.op(image.data)
        #     fourier_op = NFFT2(samples=convert_mask_to_locations(
        #                         fftshift(self.mask)), shape=image.shape)
        #     print("Process test with image '{0}'...".format(
        #         image.metadata["path"]))
        #     for nb_scale in self.nb_scales:
        #         print("- Number of scales: {0}".format(nb_scale))
        #         for name in self.names:
        #             print("    Transform: {0}".format(name))
        #             linear_op = Wavelet2(
        #                 wavelet_name=name,
        #                 nb_scale=4)
        #             gradient_op = GradAnalysis2(
        #                 data=data,
        #                 fourier_op=fourier_op)
        #             prox_dual_op = Threshold(None)
        #             x_final, transform, _, _ = sparse_rec_condatvu(
        #                 gradient_op=gradient_op,
        #                 linear_op=linear_op,
        #                 prox_dual_op=prox_dual_op,
        #                 cost_op=None,
        #                 std_est=0.0,
        #                 std_est_method="dual",
        #                 std_thr=0,
        #                 mu=0,
        #                 tau=None,
        #                 sigma=None,
        #                 relaxation_factor=1.0,
        #                 nb_of_reweights=0,
        #                 max_nb_of_iter=self.nb_iter,
        #                 add_positivity=False,
        #                 atol=1e-4,
        #                 verbose=0)
        #             mismatch = (1. - numpy.mean(
        #                 numpy.isclose(x_final, fourier.adj_op(data),
        #                               rtol=1e-3)))
        #             print("      mismatch = ", mismatch)
        #             return


if __name__ == "__main__":
    unittest.main()
