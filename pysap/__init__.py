# -*- coding: utf-8 -*-
##########################################################################
# PySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""PySAP.

This section provides full API documentation for the core PySAP package, which
handles :py:class:`Image <pysap.base.image.Image>` objects, data
:py:class:`transforms <pysap.extensions.transform>` and other key features.

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
