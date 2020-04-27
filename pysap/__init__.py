# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
pySap is a Python package related to sparsity and its application in
astronomical data analysis. This package is based on the 'sparse2d' C++ library
that allows sparse decomposition, denoising and deconvolution.
"""

# import matplotlib
# matplotlib.use("Agg")

from .info import __version__
import pysap.extensions
from pysap.base import io
from pysap.utils import wavelist
from pysap.utils import TempDir
from pysap.configure import info
from pysap.base.image import Image
from pysap.utils import load_transform
from pysap.utils import AVAILABLE_TRANSFORMS


# Display a welcome message
# print(info())
