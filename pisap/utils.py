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
    def __init__(self):
        self.path = None
        return

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, type, value, traceback):
        if self.path is not None:
            shutil.rmtree(self.path)


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
