##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common tools for MRI image reconstruction.
"""


# System import
import numpy as np


def flatten(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of ndarray or ndarray
        the input dataset.

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of uplet
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape = [x[0].shape]
    for data in x[1:]:
        y = np.concatenate((y, data.flatten()))
        shape.append(data.shape)

    return y, shape


def unflatten(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of uplet
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    offset = 0
    x = []
    for size in shape:
        start = offset
        stop = offset + np.prod(size)
        offset = stop
        x.append(y[start: stop].reshape(size))

    return x


def fista_logo():
    """ Return a nice ascii logo for the FISTA optimization using the dansing
    font.

    Returns
    -------
    logo: str
        the desired ascii logo.
    """
    logo = r"""
  _____             ____     _____      _
 |" ___|    ___    / __"| u |_ " _| U  /"\  u
U| |_  u   |_"_|  <\___ \/    | |    \/ _ \/
\|  _|/     | |    u___) |   /| |\   / ___ \\
 |_|      U/| |\u  |____/>> u |_|U  /_/   \_\\
 )(\\\,-.-,_|___|_,-.)(  (__)_// \\\_  \\\    >>
(__)(_/ \_)-' '-(_/(__)    (__) (__)(__)  (__)
    """
    return logo


def condatvu_logo():
    """ Return a nice ascii logo for the CONDAT-VU optimization using the
    dansing font.

    Returns
    -------
    logo: str
        the desired ascii logo.
    """
    logo = r"""
   ____   U  ___ u  _   _    ____       _       _____      __     __    _   _
U /"___|   \/"_ \/ | \ |"|  |  _"\  U  /"\  u  |_ " _|     \ \   /"/uU |"|u| |
\| | u     | | | |<|  \| |>/| | | |  \/ _ \/     | |        \ \ / //  \| |\| |
 | |/__.-,_| |_| |U| |\  |uU| |_| |\ / ___ \    /| |\       /\ V /_,-. | |_| |
  \____|\_)-\___/  |_| \_|  |____/ u/_/   \_\  u |_|U      U  \_/-(_/ <<\___/
 _// \\      \\    ||   \\,-.|||_    \\    >>  _// \\_       //      (__) )(
(__)(__)    (__)   (_")  (_/(__)_)  (__)  (__)(__) (__)     (__)         (__)
    """
    return logo


def convert_mask_to_locations(mask):
    """ Return the converted Cartesian mask as sampling locations.

    Parameters
    ----------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.

    Returns
    -------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    """
    row, col = np.where(mask == 1)
    row = row.astype("float") / mask.shape[0] - 0.5
    col = col.astype("float") / mask.shape[1] - 0.5
    return np.c_[row, col]


def convert_locations_to_mask(samples_locations, img_shape):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters
    ----------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    img_shape: tuple of int
        shape of the desired mask, not necessarly a square matrix.

    Returns
    -------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.
    """
    samples_locations = samples_locations.astype("float")
    samples_locations += 0.5
    samples_locations[:, 0] *= img_shape[0]
    samples_locations[:, 1] *= img_shape[1]
    samples_locations = np.floor(samples_locations)
    samples_locations = samples_locations.astype("int")
    mask = np.zeros(img_shape)
    mask[samples_locations[:, 0], samples_locations[:, 1]] = 1
    return mask
