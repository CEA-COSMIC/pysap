##########################################################################
# XXX - Copyright (C) XXX, 2017
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
import shutil
import tempfile

# Package import
import pisap.extensions.transform
from pisap.base.transform import WaveletTransformBase


AVAILABLE_TRANSFORMS = sorted(WaveletTransformBase.REGISTRY.keys())


def load_transform(name):
    """ Load a transformation using his name.

    All the available transfroms are stored in the 'pisap.AVAILABLE_TRANSFORMS'
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
    """ PISAP logo is ascii art using fender font.

    Returns
    -------
    logo: str
        the pisap logo.
    """
    logo = """
'||'''|, |''||''| .|'''|       /.\      '||'''|,
 ||   ||    ||    ||          // \\      ||   ||
 ||...|'    ||    `|'''|,    //...\\     ||...|'
 ||         ||     .   ||   //     \\    ||
.||      |..||..|  |...|' .//       \\. .||    '
                                       ''   """
    return logo
