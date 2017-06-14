##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
#:Author: Samuel Farrens <samuel.farrens@gmail.com>
#:Version: 1.1
#:Date: 06/01/2017
##########################################################################

"""
This module contains classes for reweighting optimisation implementations

**References**

1) Candes, Wakin and Boyd, Enhancing Sparsity by Reweighting l1
Minimization, 2007, Journal of Fourier Analysis and Applications,
14(5):877-905. (CWB2007)
"""

# System import
import numpy as np
from pisap.base.dictionary import DictionaryBase

class cwbReweight(object):
    """ Candes, Wakin and Boyd reweighting class

    This class implements the reweighting scheme described in CWB2007

    Parameters
    ----------
    weights: np.ndarray
        Array of weights
    thresh_factor: float (optional, default 1)
        Threshold factor
    wtype: str (optional, default 'image')
        In which domain the weights are defined: 'image' or 'sparse'.
    """
    def __init__(self, weights, thresh_factor=1, wtype="image"):
        self.weights = weights
        self.wtype = wtype
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor

    def reweight(self, data):
        """ Reweight.

        This method implements the reweighting from section 4 in CWB2007.

        Notes
        -----
        Reweighting implemented as:

        .. code::

            w = w (1 / (1 + |x^w|/(n * sigma)))
        """
        self.weights *= (1.0 / (1.0 + data.absolute / (self.thresh_factor *
                         self.original_weights)))


class mReweight(object):
    """ Ming reweighting class

    This class implements the reweighting scheme described in Ming2017

    Parameters
    ----------
    weights: np.ndarray
        Array of weights
    thresh_factor: float (optional, default 1)
        Threshold factor: sigma threshold.
    wtype: str (optional, default 'image')
        In which domain the weights are defined: 'image' or 'sparse'.
    """
    def __init__(self, weights, thresh_factor=1, wtype="image"):
        self.weights = weights
        self.wtype = wtype
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor

    def reweight(self, sigma_est, y):
        """ Reweight.

        This method implements the reweighting from section 4 in CWB2007.

        Notes
        -----
        Reweighting implemented as:

        .. code::

            w = (Ksig * sigma_i) / |Wtx_i| if |Wtx_i| > (Ksig * sigma_i)
        """
        weights = np.ones_like(y._data)
        for ks in range(y.nb_scale):
            thr = self.thresh_factor * sigma_est[ks]
            index = y.get_scale(ks) > thr
            scale_mask = y.get_scale_mask(ks)
            weights[scale_mask][index] = thr / np.abs(y.get_scale(ks)[index])
        self.weights._data = weights
