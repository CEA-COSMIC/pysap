# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy

# Package import
import pysap
from pysap.base.exceptions import Exception
from pysap.base.loaders import FITS
from pysap.base.loaders import NIFTI
from pysap.base.loaders import npBinary

# Global parameters
# > define all the available loaders
LOADERS = [FITS, NIFTI, npBinary]


def load(path, dtype=numpy.single):
    """ Load an image.

    Parameters
    ----------
    path: str
        the path to the data to be loaded.
    dtype: str
       type to which the data will be cast. Passing 'None' will not cast.

    Returns
    -------
    image: Image
        the loaded image.
    """
    # Load the image
    loader = get_loader(path)
    image = loader.load(path)

    # Cast the image if requested
    if dtype:
        image.data = image.data.astype(dtype)

    return image


def save(image, path):
    """ Save an image.

    Parameters
    ----------
    image: Image or ndarray
        the data to be saved.
    path: str
        the destination file.
    """
    # Get the data
    if not isinstance(image, pysap.Image):
        image = pysap.Image(data=image)

    # Save the data
    saver = get_saver(path)
    saver.save(image, path)


def get_loader(path):
    """ Search for a suitable loader in the declared loaders.
    Raise an exception if no loader is found.

    Parameters
    ----------
    path: str
        the path to the data to be loaded.

    Returns
    -------
    loader: @instance
        the loader instance.
    """
    for loader_class in LOADERS:
        loader = loader_class()
        if loader.can_load(path):
            return loader
    raise Exception("No loader available for '{0}'.".format(path))


def get_saver(path):
    """ Search for a suitable saver in the declared savers.
    Raise an exception if no saver is found.

    Parameters
    ----------
    path: str
        the path to the data to be saved.

    Returns
    -------
    saver: @instance
        the loader instance.
    """
    for saver_class in LOADERS:
        saver = saver_class()
        if saver.can_save(path):
            return saver
    raise Exception("No saver available for '{0}'.".format(path))
