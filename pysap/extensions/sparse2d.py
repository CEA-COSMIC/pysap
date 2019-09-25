# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import

import warnings

# Package import
import pysap

from pysap.base.transform import MetaRegister  # for the metaclass
from pysap.base import image

try:
    import pysparse
except ImportError:
    warnings.warn("Sparse2d python bindings not found, use binaries.")
    pysparse = None

# Third party import
import numpy as np
import matplotlib.pyplot as plt


class Filter():
    def __init__(self, **kwargs):
        self.data = None
        self.flt = pysparse.MRFilters(**kwargs)

    def filter(self, data):
        self.data = pysap.Image(data=self.flt.filter(data))
