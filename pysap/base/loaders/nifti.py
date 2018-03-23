##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import nibabel
import numpy

# Package import
from .loader_base import LoaderBase
from pysap.base.image import Image


class NIFTI(LoaderBase):
    """ Define the Nifti loader.
    """
    allowed_extensions = [".nii", ".nii.gz"]

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
        _image = nibabel.load(path)
        return Image(spacing=_image.header.get_zooms(),
                     data_type="scalar",
                     metadata={"path": path},
                     data=_image.get_data())

    def save(self, image, outpath):
        """ A method that save the image data and associated metadata.

        Parameters
        ----------
        image: Image
            the image to be saved.
        outpath: str
            the path where the the image will be saved.
        """
        diag = (1. / image.spacing).tolist() + [1]
        _image = nibabel.Nifti1Image(image.data, numpy.diag(diag))
        nibabel.save(_image, outpath)
