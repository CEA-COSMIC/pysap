# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains shortcuts to utility functions.
"""

# Package import
from pysap.plugins.mri.reconstruct.utils import generate_operators
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.reconstruct.utils import convert_locations_to_mask

import numpy as np


def normalize_samples(samples):
    """
    This function normalize the samples so it can be between [-0.5; 0.5[ for the
    non-cartesian case
    Parameters:
    -----------
        samples: np.ndarray
            Unnormalized samples
    Return:
    -------
        normalized_samples: np.ndarray
            Same shape as the parameters but with values between [-0.5; 0.5[
    """
    samples_locations = np.copy(samples.astype('float'))
    samples_locations[:, 0] /= 2 * np.abs(samples_locations[:, 0]).max()
    samples_locations[:, 1] /= 2 * np.abs(samples_locations[:, 1]).max()
    while samples_locations.max() == 0.5:
        dim1, dim2 = np.where(samples_locations == 0.5)
        samples_locations[dim1, dim2] = -0.5
    return samples_locations
