# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module with usefull plottinting utility functions.
"""

# Package import
import pisap

# Third party import
import numpy as np
from skimage import data, img_as_float
from skimage import exposure


def scaling(image, method="stretching"):
    """
    Change the image dynamic.

    Parameters
    ----------
    image: Image
        the image to be transformed.
    method: str, default 'stretching'
        the normalization method: 'stretching', 'equalization' or 'adaptive'.

    Returns
    -------
    normalize_image: Image
        the normalized image.
    """
    # Contrast stretching
    if method == "stretching":
        p2, p98 = np.percentile(image.data, (2, 98))
        norm_data = exposure.rescale_intensity(image.data, in_range=(p2, p98))

    # Equalization
    elif method == "equalization":
        norm_data = exposure.equalize_hist(image.data)

    # Adaptive Equalization
    elif method == "adaptive":
        norm_data = exposure.equalize_adapthist(image.data, clip_limit=0.03)

    # Unknown method
    else:
        raise ValueError("Unknown normalization '{0}'.".format(method))

    normalize_image = pisap.Image(data=norm_data)

    return normalize_image
