##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
piSAP is a Python package related to sparsity and its application in
astronomical data analysis. This package is based on the 'sparse2d' C++ library
that allows sparse decomposition, denoising and deconvolution.
"""

from .info import __version__
from pisap.base import io
from pisap.base.image import Image
import pisap.extensions

