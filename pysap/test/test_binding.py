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
import os
import numpy
import sys
import time

# Package import
import pysap
import pysap.extensions.transform
from pysap.data import get_sample_data


class TestWarpAndBinding(unittest.TestCase):
    """ Test the analysis/synthesis of an input image using a wrapping or a
    binding strategy.
    """
    def setUp(self):
        """ Get the data from the server.
        """
        self.images = [
            # get_sample_data(dataset_name="astro-fits"),
            get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [i.data.shape for i in self.images]))
        self.transforms = [
            pysap.load_transform(name) for name in pysap.AVAILABLE_TRANSFORMS]
        print("[info] Found {0} transformations.".format(len(self.transforms)))
        self.nb_scales = [3]  # [2, 3, 4]
        self.nb_iter = 10

    def test_wavelet_transformations(self):
        """ Test all the registered transformations.
        """
        for image in self.images:
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for transform in self.transforms:
                    print("    Transform: {0}".format(transform))
                    transform = transform(nb_scale=nb_scale, verbose=0)
                    self.assertFalse(transform.use_wrapping)
                    transform.info
                    transform.data = image
                    transform.analysis()
                    # transform.show()
                    recim = transform.synthesis()
                    # recim.show()
                    mismatch = (1. - numpy.mean(
                        numpy.isclose(recim.data, image.data, atol=1e-8,
                                      rtol=1e-5)))
                    print("      mismatch = ", mismatch)
                    print("      analysis = ",
                          [band.shape for band in transform.analysis_data],
                          len(transform.analysis_data))
                    print("      bands = ", transform.nb_band_per_scale)
                    print("      synthesis = ", recim.shape)

    def test_speed(self):
        """ Test the bindings time advantages.
        """
        # With/without bindings
        for strategy, name in ((True, "Without"), (False, "With")):
            tic = time.time()
            transform = pysap.load_transform(
                "LinearWaveletTransformATrousAlgorithm")
            transform = transform(nb_scale=4, verbose=0)
            transform.use_wrapping = strategy
            transform.data = self.images[0]
            for i in range(self.nb_iter):
                transform.analysis()
                recim = transform.synthesis()
            toc = time.time()
            print("[result] {0} bindings execution time: {1}.".format(
                name, toc - tic))

    def test_accessors(self):
        """ Test all the accessors.
        """
        # With/without bindings
        for strategy, name in ((True, "without"), (False, "with")):

            # Test 3-bands undecimated transform
            nb_scale = 4
            print("[info] Test {0} bindings.".format(name))
            transform = pysap.load_transform(
                "NonOrthogonalUndecimatedTransform")
            transform = transform(nb_scale=nb_scale, verbose=0)
            transform.use_wrapping = strategy
            transform.data = self.images[0]
            transform.analysis()

            # Get with scale index only
            for scale in range(nb_scale - 1):
                band_data = transform[scale]
                self.assertEqual(len(band_data), 3)
                for band_array in band_data:
                    self.assertEqual(band_array.shape, (512, 512))
            band_array = transform[nb_scale - 1]
            self.assertEqual(band_array.shape, (512, 512))

            # Get with scale and band
            self.assertEqual(transform[0, 0].shape, (512, 512))

            # Get with scale and band as slice
            band_data = transform[2, 1:3:1]
            self.assertEqual(len(band_data), 2)
            for band_array in band_data:
                self.assertEqual(band_array.shape, (512, 512))

            # Get with scale as slice and band
            band_data = transform[1:3, 0]
            self.assertEqual(len(band_data), 2)
            for band_array in band_data:
                self.assertEqual(band_array.shape, (512, 512))

            # Modify a band on the fly
            band_array = transform[0, 0]
            band_array[:, :] = 10
            self.assertTrue(numpy.allclose(transform[0, 0], band_array))


if __name__ == "__main__":
    unittest.main()
