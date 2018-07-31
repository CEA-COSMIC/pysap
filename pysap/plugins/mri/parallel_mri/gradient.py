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
from .utils import check_lipschitz_cst

# Third party import
import numpy as np
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class Gradient_pMRI_analysis(GradBasic, PowerMethod):
    """ Gradient analysis class.

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
    def __init__(self, data, fourier_op, S=None):
        """ Initilize the 'GradSynthesis' class.
        """

        self.fourier_op = fourier_op
        self.p_MRI = False if S is None else True
        if self.p_MRI:
            self.smaps = S
        GradBasic.__init__(self, data, self._analy_op_method,
                           self._analy_rsns_op_method)
        PowerMethod.__init__(self, self.trans_op_op, self.fourier_op.shape,
                             data_type="complex128", auto_run=False)
        self.get_spec_rad(extra_factor=1.1, max_iter=5)

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
        if self.p_MRI:
            kspace = [self.fourier_op.op(self.smaps[channel] * x) for
                      channel in range(self.smaps.shape[0])]
            return np.asarray(kspace)

        else:
            return self.fourier_op.op(x)

    def _analy_rsns_op_method(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input kspace nD array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        if self.p_MRI:
            y = [np.conj(self.smaps[channel]) * self.fourier_op.adj_op(
                 x[channel]) for channel in range(self.smaps.shape[0])]
            y = np.asarray(y)
            return np.squeeze(np.sum(y, axis=0))

        else:
            return self.fourier_op.adj_op(x)


class Gradient_pMRI_synthesis(GradBasic, PowerMethod):
    """ Gradient synthesis class.

    This class defines the grad operators for |M*F*invL*alpha - data|**2.

    Parameters
    ----------
    data: np.ndarray
        input nD data array.
    fourier_op: instance
        a Fourier operator instance.
    linear_op: object
        a linear operator instance.
    S: np.ndarray  (L, image_shape)
        The sensitivity maps of size.
    """
    def __init__(self, data, fourier_op, linear_op, S=None):
        """ Initilize the 'GradSynthesis2' class.
        """

        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.p_MRI = True
        if S is not None:
            self.smaps = S
        else:
            self.p_MRI = False

        GradBasic.__init__(self, data, self._synth_op_method,
                           self._synth_trans_op_method)
        coef = linear_op.op(np.zeros(fourier_op.shape).astype(np.complex))
        self.linear_op_coeffs_shape = coef.shape
        PowerMethod.__init__(self, self.trans_op_op, coef.shape,
                             data_type="complex128", auto_run=False)
        self.get_spec_rad(extra_factor=1.1, max_iter=5)

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
        img = self.linear_op.adj_op(x)
        if self.p_MRI:
            rsl = np.asarray([self.fourier_op.op(self.smaps[l] * img) for l
                              in range(self.smaps.shape[0])])
        else:
            rsl = self.fourier_op.op(img)
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
        if self.p_MRI:
            tmp = [self.fourier_op.adj_op(x[l]) for l in
                   range(self.smaps.shape[0])]
            rsl = np.sum([self.linear_op.op(tmp[l] * np.conj(self.smaps[l]))
                          for l in range(self.smaps.shape[0])], axis=0)
        else:
            rsl = self.linear_op.op(self.fourier_op.adj_op(x))
        return rsl


class Gradient_pMRI(Gradient_pMRI_analysis, Gradient_pMRI_synthesis):
    """ Gradient for parallel imaging reconstruction.

    This class defines the datafidelity terms methods that will be defined by
    derived gradient classes:
    It computes the gradient of the following equation for the analysis and
    synthesis cases respectively:

    * (1/2) * sum(||Ft Sl x - yl||^2_2,l)
    * (1/2) * sum(||Ft Sl L* alpha - yl||^2_2,l)
    """
    def __init__(self, data, fourier_op, S=None, linear_op=None,
                 check_lips=False):
        """ Initilize the 'Gradient_pMRI' class.

        Parameters
        ----------
        data: np.ndarray
            input nd data array.
        fourier_op: instance
            a Fourier operator instance derived from the FourierBase' class.
        S: np.ndarray
            The sensitivity matrix shape [L, img_shape]
            for 2D images:[L, N1, N2]; for 3D images: [L, N1, N2, N3]
        linear_op: instance
            a Linear operator instance.
        check_lips: boolean
            Check if the calculated Lipschitz constant satisfies the constaints
        """
        if linear_op is None:
            Gradient_pMRI_analysis.__init__(self, data, fourier_op, S)
            if check_lips:
                xinit_shape = fourier_op.img_shape
            self.analysis = True
        else:
            Gradient_pMRI_synthesis.__init__(self, data, fourier_op, linear_op,
                                             S)
            if check_lips:
                xinit_shape = self.linear_op_coeffs_shape
            self.analysis = False

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
