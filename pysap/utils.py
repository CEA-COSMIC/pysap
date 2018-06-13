# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module with common functions to list and load transformations.
"""

# System import
from __future__ import division, print_function, absolute_import
import os
import shutil
import tempfile

# Package import
import pysap.extensions.transform
from pysap.base.transform import WaveletTransformBase


AVAILABLE_TRANSFORMS = sorted(WaveletTransformBase.REGISTRY.keys())


def load_image(image_path):
    """
    Load an image.

    Parameters
    ----------
    image_path: str
        the image path.

    Returns
    -------
    image: Image
        the loaded image.
    """
    image = pysap.io.load(image_path)
    return image


def save_image(image, image_path):
    """
    Save an image.

    Parameters
    ----------
    image: Image
        the image to be saved.
    image_path: str
        the image path.
    """
    pysap.io.save(image, image_path)


def load_transform(name):
    """ Load a transformation using his name.

    All the available transfroms are stored in the 'pysap.AVAILABLE_TRANSFORMS'
    parameter.

    Parameters
    ----------
    name: str
        the name of the desired transformation.

    Returns
    -------
    transform: WaveletTransformBase
        the transformation.
    """
    if name not in WaveletTransformBase.REGISTRY:
        raise ValueError("Unknown transform '{0}'. Allowed transforms are: "
                         "{1}.".format(name, AVAILABLE_TRANSFORMS))
    return WaveletTransformBase.REGISTRY[name]


class TempDir(object):
    """ Create a tempdir with the with synthax.
    """
    def __init__(self, isap=False):
        """ Initialize the TempDir class.

        Parameters
        ----------
        isap: bool, default False
            if set, generates a temporary folder compatible with ISAP.
        """
        self.path = None
        self.isap = isap
        return

    def __enter__(self):
        if self.isap:
            self.path = self._mkdtemp_isap()
        else:
            self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, type, value, traceback):
        if self.path is not None:
            shutil.rmtree(self.path)

    def _mkdtemp_isap(self):
        """ Method to generate a temporary folder compatible with the ISAP
        implementation.

        If 'jpg' or 'pgm' (with any case for each letter) are in the pathname,
        it will corrupt the format detection in ISAP.

        Returns
        -------
        tmpdir: str
            the generated ISAP compliant temporary folder.
        """
        tmpdir = None
        while (tmpdir is None or "pgm" in tmpdir.lower() or
               "jpg" in tmpdir.lower()):
            if tmpdir is not None and os.path.exists(tmpdir):
                os.rmdir(tmpdir)
            tmpdir = tempfile.mkdtemp()
        return tmpdir


def logo():
    """ pySAP logo is ascii art using fender font.

    Returns
    -------
    logo: str
        the pysap logo.
    """
    logo = """
                 .|'''|       /.\      '||'''|,
                 ||          // \\      ||   ||
'||''|, '||  ||` `|'''|,    //...\\     ||...|'
 ||  ||  `|..||   .   ||   //     \\    ||
 ||..|'      ||   |...|' .//       \\. .||
 ||       ,  |'
.||        ''"""
    return logo


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
