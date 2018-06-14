# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from scipy.io import loadmat, savemat
import numpy

# Package import
from .loader_base import LoaderBase
from pysap.base.image import Image
from pysap.base.exceptions import Exception


class MAT(LoaderBase):
    """ Define the Fits loader.
    """
    allowed_extensions = [".mat"]

    def load(self, path):
        """ A method that load the image data and associated metadata.

        Parameters
        ----------
        path: str
            the path to the image to be loaded.

        Return
        ------
        image: Image
            the loaded image.
        """
        cube = loadmat(path)['samples']  # ## TODO: Make it more general
        return Image(data_type="scalar",
                     metadata={"path": path},
                     data=cube)

    def save(self, image, outpath, clobber=True):
        """ A method that save the image data and associated metadata.

        Parameters
        ----------
        image: Image
            the image to be saved.
        outpath: str
            the path where the the image will be saved.
        clobber: bool (optional, default True)
            If True, and if filename already exists, it will overwrite the
            file.
        """
        savemat({'samples': image.data})
