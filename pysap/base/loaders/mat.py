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
from scipy.io import loadmat, savemat
import numpy


class MAT(LoaderBase):
    """ Define the '.mat' file loader.
    """
    allowed_extensions = [".mat"]

    def load(self, path, image_field="image", meta_field="metadata"):
        """ A method that load the data and associated metadata.

        Parameters
        ----------
        path: str
            the path to the data to be loaded.
        image_field: str, default 'image'
            the name of the data field that contains the image array.
        image_field: str, default 'metadata'
            the name of the data field that contains the image metadata.

        Returns
        -------
        image: Image
            the loaded image.
        """
        data = loadmat(path)
        _array = data[image_field]
        _meta = {"path": path}
        if meta_field in data:
            _meta.update(data[meta_field])
        return Image(data_type="scalar",
                     metadata=_meta,
                     data=_array)

    def save(self, image, outpath, image_field="image", meta_field="metadata"):
        """ A method that save the image and associated metadata.

        Parameters
        ----------
        image: Image
            the image to be saved.
        outpath: str
            the path where the the image will be saved.
        image_field: str, default 'image'
            the name of the data field that contains the image array.
        image_field: str, default 'metadata'
            the name of the data field that contains the image metadata.
        """
        data = {
            image_field: image.data,
            meta_field: image.metadata}
        savemat(data)
