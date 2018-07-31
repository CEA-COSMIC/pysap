# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Different cost functions for the optimization.
"""


# Third party import
import numpy
from modopt.opt.cost import costObj


class GenericCost(costObj):
    """ Define the Generic cost function, based on the get_cost function of the
    gradient operator and the get_cost function of the proximity operator.
    """
    def __init__(self, gradient_op, prox_op, linear_op=None, initial_cost=1e6,
                 tolerance=1e-4, cost_interval=1, test_range=4, verbose=True,
                 plot_output=None):
        """ Initialize the 'LassoCost' class.

        Parameters:
        -----------
        gradient_op: instance of the gradient operator
            gradient operator used in the reconstruction process. It must
            implements the get_cost_function.
        prox_op: instance of the proximity operator
            proximity operator used in the reconstruction process. It must
            implements teh get_cost function.
        linear_op: instance of the linear operator
            linear operator used to express the sparsity.
            If the synthesis formulation is used to solve the problem than the
            parameter has to be set to 0.
            If the analysis formultaion is used to solve the problem than the
            parameters needs to be filled
        initial_cost: float, optional
            Initial value of the cost (default is "1e6")
        tolerance: float, optional
            Tolerance threshold for convergence (default is "1e-4")
        cost_interval: int, optional
            Iteration interval to calculate cost (default is "1")
        test_range: int, optional
            Number of cost values to be used in test (default is "4")
        verbose: bool, optional
            Option for verbose output (default is "True")
        plot_output: str, optional
            Output file name for cost function plot
        """
        gradient_cost = getattr(gradient_op, 'get_cost', None)
        prox_cost = getattr(prox_op, 'get_cost', None)
        if not callable(gradient_cost):
            raise RuntimeError("The gradient must implements a `get_cost`",
                               "function")
        if not callable(prox_cost):
            raise RuntimeError("The proximity operator must implements a",
                               " `get_cost` function")
        self.gradient_op = gradient_op
        self.prox_op = prox_op
        self.analysis = True
        self.linear_op = linear_op
        if hasattr(gradient_op, 'linear_op'):
            self.analysis = False
            self.linear_op = gradient_op.linear_op

        if self.linear_op is None:
            raise ValueError("The analysis formulation has been detected, the",
                             "linear operator must be filled")

        super(GenericCost, self).__init__(
            operators=None, initial_cost=initial_cost,
            tolerance=tolerance,
            cost_interval=cost_interval, test_range=test_range,
            verbose=verbose, plot_output=plot_output)
        self._iteration = 0

    def _calc_cost(self, x_new, *args, **kwargs):
        """ Return the Lasso cost.

        Parameters
        ----------
        x_new: np.ndarray
            intermediate solution in the optimization problem.

        Returns
        -------
        cost: float
            the cost function defined by the operators (gradient + prox_op).
        """
        if self.analysis:
            cost = self.gradient_op.get_cost(x_new) + self.prox_op.get_cost(
                self.linear_op.op(x_new))
        else:
            cost = self.gradient_op.get_cost(x_new) + self.prox_op.get_cost(
                x_new)
        return cost
