# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
from pyqtgraph.Qt import QtGui
import pyqtgraph


def plot_data(data, scroll_axis=2):
    """ Plot an image associated data.
    Currently support on 1D, 2D or 3D data.

    Parameters
    ----------
    data: array
        the data to be displayed.
    scroll_axis: int (optional, default 2)
        the scroll axis for 3d data.
    """
    # Check input parameters
    if data.ndim not in range(1, 4):
        raise ValueError("Unsupported data dimension.")

    # Deal with complex data
    if numpy.iscomplex(data).any():
        data = numpy.abs(data)

    # Create application
    app = pyqtgraph.mkQApp()

    # Create the widget
    if data.ndim == 3:
        indices = [i for i in range(3) if i != scroll_axis]
        indices = [scroll_axis] + indices
        widget = pyqtgraph.image(numpy.transpose(data, indices))
    elif data.ndim == 2:
        widget = pyqtgraph.image(data)
    else:
        widget = pyqtgraph.plot(data)

    # Run application
    app.exec_()
