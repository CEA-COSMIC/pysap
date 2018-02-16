# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains a wrapping of the 'sparse2d' C++ library
that allows fast sparse decomposition, denoising and deconvolution.
"""

from .tools import mr_transform
from .tools import mr_filter
from .tools import mr_deconv
from .tools import mr_recons
from .formating import FLATTENING_FCTS as ISAP_FLATTEN
from .formating import INFLATING_FCTS as ISAP_UNFLATTEN
