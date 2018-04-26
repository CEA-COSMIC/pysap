# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np

# Package import
from .loader_base import LoaderBase
from pysap.base.image import Image


class npBinary(LoaderBase):
    """ Define the numpy binary loader.
    """
    allowed_extensions = [".npy"]

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

        cube = np.load(path)
        return Image(data_type="scalar",
                     data=cube)

    def save(self, image, outpath, clobber=True):
        """ A method that save the image data and associated metadata.

        Parameters
        ----------
        image: Image
            the image to be saved.
        outpath: str
            the path where the the image will be saved.
        """

        np.save(outpath, image.data)
