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

try:
    import pyqtgraph
    from pyqtgraph.Qt import QtGui
except ImportError:  # pragma: no cover
    pyqt_found = False
else:
    pyqt_found = True



def plot_data(data, scroll_axis=2):
    """ Plot an image associated data.
    Currently support on 1D, 2D or 3D data.

    Parameters
    ----------
    data: numpy.ndarray
        the data to be displayed.
    scroll_axis: int (optional, default 2)
        the scroll axis for 3d data.

    Notes
    -----
    This function is deprecated and will be removed in a future release.

    """
    if not pyqt_found:
        raise ImportError(
            'To use this deprecated function you will need to install '
            + 'pyqtgraph manually. Note that the current implementation '
            + 'is not compatible with PyQT v6.'
        )

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
