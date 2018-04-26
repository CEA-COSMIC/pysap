# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


class LoaderBase(object):
    """ Base class for all loaders.
    """
    allowed_extensions = []

    def can_load(self, path):
        """ A method checking the file extension.

        Parameters
        ----------
        path: str
            the path to the image to be loaded.

        Return
        ------
        out: bool
            True if the file extension is valid, False otherwise.
        """
        for ext in self.allowed_extensions:
            if path.endswith(ext):
                return True
        return False

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
        raise NotImplementedError(
            "The 'load' method must be implemented in subclasses.")

    def can_save(self, outpath):
        """ A method checking the output file extension.

        Parameters
        ----------
        outpath: str
            the path where the the image will be saved.

        Return
        ------
        out: bool
            True if the output file extension is valid, False otherwise.
        """
        for ext in self.allowed_extensions:
            if outpath.endswith(ext):
                return True
        return False

    def save(self, image, outpath):
        """ A method that save the image data and associated metadata.

        Parameters
        ----------
        image: Image
            the image to be saved.
        outpath: str
            the path where the the image will be saved.
        """
        raise NotImplementedError(
            "The 'save' method must be implemented in subclasses.")
