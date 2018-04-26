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
# import os
import numpy
from scipy.fftpack import fftshift
# import sys
# import time

# Package import
import pysap
from pysap.plugins.mri.reconstruct.fourier import FFT2
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct.gradient import GradAnalysis2
from pysap.plugins.mri.reconstruct.gradient import GradSynthesis2
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
import pysap.extensions.transform
from pysap.data import get_sample_data


class TestOptimizer(unittest.TestCase):
    """ Test the FISTA's gradient descent.
    """
    def setUp(self):
        """ Get the data from the server.
        """
        self.images = [
            # get_sample_data(dataset_name="astro-fits"),
            get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [i.data.shape for i in self.images]))
        self.mask = get_sample_data("mri-mask").data
        self.names = ['BsplineWaveletTransformATrousAlgorithm']
        print("[info] Found {0} transformations.".format(len(self.names)))
        self.nb_scales = [4]  # [2, 3, 4]
        self.nb_iter = 300

    def test_reconstruction_fista_fft2(self):
        """ Test all the registered transformations.
        """
        print('Process test FFT2 FISTA')
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
                    linear_op = Wavelet2(wavelet_name=name,
                                         nb_scale=4)
                    gradient_op = GradSynthesis2(data=data,
                                                 fourier_op=fourier_op,
                                                 linear_op=linear_op)
                    x_final, transform = sparse_rec_fista(
                                            gradient_op=gradient_op,
                                            linear_op=linear_op,
                                            mu=0,
                                            lambda_init=1.0,
                                            max_nb_of_iter=self.nb_iter,
                                            atol=1e-4,
                                            verbose=0,
                                            get_cost=False)

                    mismatch = (1. - numpy.mean(
                        numpy.isclose(x_final, fourier.adj_op(data),
                                      rtol=1e-3)))
                    print("      mismatch = ", mismatch)

    def test_reconstruction_condat_vu_fft2(self):
        """ Test all the registered transformations.
        """
        print('Process test FFT2 Condat Vu algorithm')
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
                    linear_op = Wavelet2(wavelet_name=name,
                                         nb_scale=4)
                    gradient_op = GradAnalysis2(data=data,
                                                fourier_op=fourier_op)
                    x_final, transform = sparse_rec_condatvu(
                                            gradient_op=gradient_op,
                                            linear_op=linear_op,
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

                    mismatch = (1. - numpy.mean(
                        numpy.isclose(x_final, fourier.adj_op(data),
                                      rtol=1e-3)))
                    print("      mismatch = ", mismatch)
                    return


if __name__ == "__main__":
    unittest.main()
