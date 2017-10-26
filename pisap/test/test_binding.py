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
import os
import numpy
import sys
import time

# Package import
import pisap
import pisap.extensions.transform
from pisap.data import get_sample_data
from pisap.base.transform import WaveletTransformBase


class TestAnalysisSynthesis(unittest.TestCase):
    """ Test the analysis/synthesis of an input image.
    """
    def setUp(self):
        """ Get the data from the server.
        """
        fits_file = get_sample_data(dataset_name="astro-fits")
        nii_file = get_sample_data(dataset_name="mri-slice-nifti")
        self.images = [
            pisap.io.load(fits_file),
            pisap.io.load(nii_file)]
        print("[info] Image loaded for test: {0}.".format(
            [i.data.shape for i in self.images]))
        self.transforms = WaveletTransformBase.REGISTRY.values()
        print("[info] Found {0} transformations.".format(len(self.transforms)))
        self.nb_scales = [2, 3, 4]
        self.nb_iter = 1000

    def test_wavelet_transformations(self):
        """ Test all the registered transformations.
        """
        for image in self.images:
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for transform in self.transforms:
                    if transform.__isap_transform_id__ != 1:
                        continue
                    print("    Transform: {0}".format(transform))
                    transform = transform(nb_scale=nb_scale, verbose=1)
                    transform.data = image
                    transform.analysis()
                    #print([arr.shape for arr in transform.analysis_data])
                    # transform.show()
                    recim = transform.synthesis()
                    # recim.show()
                    #mismatch = (1. - numpy.mean(
                    #    numpy.isclose(recim.data, image.data, atol=1e-8,
                    #                  rtol=1e-5)))
                    #print(mismatch)

    def test_speed(self):
        """ Test the bindings time advantages.
        """
        tic = time.time()
        transform = WaveletTransformBase.REGISTRY["LinearWaveletTransformATrousAlgorithm"](
            nb_scale=4, verbose=0)
        transform.data = self.images[0]
        for i in range(self.nb_iter):
            transform.analysis()
            recim = transform.synthesis()
        toc = time.time()
        print("[result] Execution time: {0}.".format(toc - tic))



if __name__ == "__main__":
    unittest.main()
