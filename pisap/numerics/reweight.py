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
        if not isinstance(y, DictionaryBase): # to preserve code legacy
            ycube = y.to_cube()
            weights = np.ones_like(ycube)
            nb_scales = ycube.shape[0]
            for scale_index in range(nb_scales - 1):
                thr = self.thresh_factor * sigma_est[scale_index]
                index = ycube[scale_index] > thr
                weights[scale_index][index] = thr / np.abs(ycube[scale_index][index])
            self.weights.from_cube(weights[:, 0])
        else:
            weights = np.ones_like(y._data)
            for scale_index, scale_data in enumerate(y):
                thr = self.thresh_factor * sigma_est[scale_index]
                index = scale_data > thr
                scale_mask = y.get_scale_mask(scale_index)
                weights[scale_mask][index] = thr / np.abs(y.get_scale(scale_index)[index])
            self.weights._data = weights
