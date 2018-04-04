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


class DualGapCost(costObj):
    """ Define the dual-gap cost function.
    """
    def __init__(self, linear_op, initial_cost=1e6, tolerance=1e-4,
                 cost_interval=1, test_range=4, verbose=True,
                 plot_output=None):
        """ Initialize the 'DualGapCost' class.

        Parameters:
        -----------
        x: np.ndarray
            input original data array.
        costFunc: class
            Class for calculating the cost
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
        self.linear_op = linear_op
        super(DualGapCost, self).__init__(
            operators=None, initial_cost=initial_cost,
            tolerance=tolerance,
            cost_interval=cost_interval, test_range=test_range,
            verbose=verbose, plot_output=plot_output)
        self._iteration = 0

    def _calc_cost(self, x_new, y_new, *args, **kwargs):
        """ Return the dual-gap cost.

        Parameters
        ----------
        x_new: np.ndarray
            new primal solution.
        y_new: np.ndarray
            new dual solution.

        Returns
        -------
        norm: float
            the dual-gap.
        """
        x_dual_new = self.linear_op.adj_op(y_new)
        return numpy.linalg.norm(x_new - x_dual_new)
