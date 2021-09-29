# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Package import
from .loader_base import LoaderBase
from pysap.base.image import Image

# Third party import
from skimage.io import imread, imsave


class TIF(LoaderBase):
    """ Define the '.tif' file loader.
    """
    allowed_extensions = [".tif"]

    def load(self, path, as_gray=False):
        """ A method that load the data in TIF format.

        Parameters
        ----------
        path: str
            the path to the data to be loaded.
        meta_field: str, default 'metadata'
            

        Returns
        -------
        image: Image
            the loaded image.
        """
        _array = imread(path, as_gray)
        _meta = {"path": path}
        return Image(data_type="scalar",
                     metadata=_meta,
                     data=_array)

    def save(self, image, outpath, check_contrast=True):
        """ A method that save the image in TIF format.

        Parameters
        ----------
        image: Image
            the image to be saved.
        outpath: str
            the path where the the image will be saved.
        """
        imsave(outpath, image.data, check_contrast)
