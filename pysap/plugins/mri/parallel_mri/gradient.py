# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classes for defining algorithm operators and gradients.
"""

# Package import
from .utils import prod_over_maps
from .utils import function_over_maps
from .utils import check_lipschitz_cst

# Third party import
import numpy as np
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class Grad2D_pMRI_analysis(GradBasic, PowerMethod):
    """ Gradient 2D synthesis class.

    This class defines the grad operators for:
            (1/2) * sum(||Ft Sl x - yl||^2_2,l)

    Parameters
    ----------
    data: np.ndarray
        input 2D data array.
    fourier_op: instance
        a Fourier operator instance.
    S: np.ndarray
        sensitivity matrix
    """
    def __init__(self, data, fourier_op, S):
        """ Initilize the 'GradSynthesis2' class.
        """

        self.fourier_op = fourier_op
        self.S = S
        GradBasic.__init__(self, data, self._analy_op_method,
                           self._analy_rsns_op_method)
        PowerMethod.__init__(self, self.trans_op_op, self.fourier_op.shape,
                             data_type="complex128", auto_run=False)
        self.get_spec_rad(extra_factor=1.1)

    def _analy_op_method(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: a WaveletTransformBase derived object
            the analysis coefficients.

        Returns
        -------
        result: np.ndarray
            the operation result (the recovered kspace).
        """
        return function_over_maps(self.fourier_op.op,
                                  prod_over_maps(self.S, x))

    def _analy_rsns_op_method(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input kspace 2D array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        y = function_over_maps(self.fourier_op.adj_op, x)
        return np.sum(prod_over_maps(y, np.conj(self.S)), axis=2)


class Grad2D_pMRI_synthesis(GradBasic, PowerMethod):
    """ Gradient 2D synthesis class.

    This class defines the grad operators for |M*F*invL*alpha - data|**2.

    Parameters
    ----------
    data: np.ndarray
        input 2D data array.
    fourier_op: instance
        a Fourier operator instance.
    linear_op: object
        a linear operator instance.
    S: np.ndarray  (image_shape, L)
        The sensitivity maps of size.
    """
    def __init__(self, data, fourier_op, linear_op, S):
        """ Initilize the 'GradSynthesis2' class.
        """

        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.S = S
        GradBasic.__init__(self, data, self._synth_op_method,
                           self._synth_trans_op_method)
        coef = linear_op.op(np.zeros(fourier_op.shape).astype(np.complex))
        self.linear_op_coeffs_shape = coef.shape
        PowerMethod.__init__(self, self.trans_op_op, coef.shape,
                             data_type="complex128", auto_run=False)
        self.get_spec_rad(extra_factor=1.1)

    def _synth_op_method(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: a WaveletTransformBase derived object
            the analysis coefficients.

        Returns
        -------
        result: np.ndarray
            the operation result (the recovered kspace).
        """

        rsl = np.zeros(self.obs_data.shape).astype('complex128')
        img = self.linear_op.adj_op(x)
        for l in range(self.S.shape[2]):
            rsl[:, :, l] = self.fourier_op.op(self.S[:, :, l] * img)
        return rsl

    def _synth_trans_op_method(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input kspace 2D array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """

        rsl = np.zeros(self.linear_op_coeffs_shape).astype('complex128')
        for l in range(self.S.shape[2]):
            tmp = self.fourier_op.adj_op(x[:, :, l])
            rsl += self.linear_op.op(tmp *
                                     np.conj(self.S[:, :, l]))
        return rsl


class Grad2D_pMRI(Grad2D_pMRI_analysis, Grad2D_pMRI_synthesis):
    """ Gradient for 2D parallel imaging reconstruction.

    This class defines the datafidelity terms methods that will be defined by
    derived gradient classes:
    It computes the gradient of the following equation for the analysis and
    synthesis cases respectively:

    * (1/2) * sum(||Ft Sl x - yl||^2_2,l)
    * (1/2) * sum(||Ft Sl L* alpha - yl||^2_2,l)
    """
    def __init__(self, data, fourier_op, S, linear_op=None, check_lips=False):
        """ Initilize the 'Grad2D_pMRI' class.

        Parameters
        ----------
        data: np.ndarray
            input 2D data array.
        fourier_op: instance
            a Fourier operator instance derived from the FourierBase' class.
        linear_op: instance
            a Linear operator instance.
        """
        if S.shape[:2] != fourier_op.shape:
            raise ValueError('Matrix dimension not aligned')

        if linear_op is None:
            Grad2D_pMRI_analysis.__init__(self, data, fourier_op, S)
            if check_lips:
                xinit_shape = fourier_op.img_shape
            self.analysis = True
        else:
            Grad2D_pMRI_synthesis.__init__(self, data, fourier_op, linear_op,
                                           S)
            if check_lips:
                xinit_shape = self.linear_op_coeffs_shape

            self.synthesis = False

        if check_lips:
            is_lips = check_lipschitz_cst(f=self.trans_op_op,
                                          x_shape=xinit_shape,
                                          lipschitz_cst=self.spec_rad,
                                          max_nb_of_iter=10)
            if not is_lips:
                raise ValueError('The lipschitz constraint is not satisfied')
            else:
                print('The lipschitz constraint is satisfied')

    def get_cost(self, x):
        """ Gettng the cost.

        This method calculates the cost function of the differentiable part of
        the objective function.

        Parameters
        ----------
        x: np.ndarray
        input 2D data array.

        Returns
        -------
        result: float
        The result of the differentiablepart.
        """
        return 0.5 * (np.abs(self.op(x).flatten() -
                      self.obs_data.flatten())**2).sum()
