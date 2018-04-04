# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import astropy.io.fits as pyfits
import numpy

# Package import
from .loader_base import LoaderBase
from pysap.base.image import Image
from pysap.base.exceptions import Exception


class FITS(LoaderBase):
    """ Define the Fits loader.
    """
    allowed_extensions = [".fits", ".mr"]

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
        hdulist = pyfits.open(path)
        if len(hdulist) != 1:
            raise Exception("Only one HDU object supported yet. Can't "
                            "read '{0}'.".format(path))
        cube = hdulist[0].data
        header = dict(hdulist[0].header.items())
        header["path"] = path
        hdulist.close()
        return Image(data_type="scalar",
                     metadata=header,
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
        header = None
        if len(image.metadata) != 0:
            header = pyfits.Header(image.metadata.items())
        hdu = pyfits.PrimaryHDU(image.data, header=header)
        hdulist = pyfits.HDUList([hdu])
        hdulist.writeto(outpath, clobber=clobber)
