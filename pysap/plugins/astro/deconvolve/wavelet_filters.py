# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains methods for getting wavelet transform filters.
"""

import numpy as np
from pysap import load_transform


def get_cospy_filters(data_shape, transform_name, n_scales=4, coarse=False):
    """Get cospy transform filters

    This method obtains wavelet filters by calling cospy

    Parameters
    ----------
    data_shape : tuple
        2D data shape
    transform_name : str
        Name of wavelet transform
    n_scales : int, optional
        Number of transform scales (default is 4)
    coarse : bool, optional
        Option to keep coarse scale (default is 'False')

    Returns
    -------
    np.ndarray 3D array of wavelet filters

    """

    # Adjust the shape of the input data.
    data_shape = np.array(data_shape)
    data_shape += data_shape % 2 - 1

    # Create fake data.
    fake_data = np.zeros(data_shape)
    fake_data[list(zip(data_shape // 2))] = 1

    # Transform fake data
    wavelet_transform = (load_transform(transform_name)
                         (nb_scale=n_scales, verbose=True))
    wavelet_transform.data = fake_data
    wavelet_transform.analysis()
    filters = np.array(wavelet_transform.analysis_data)

    # Return filters
    if coarse:
        return filters
    else:
        return filters[:-1]
