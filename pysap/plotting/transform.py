# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import matplotlib.pyplot as plt
import numpy

try:
    import pyqtgraph
    from pyqtgraph.Qt import QtGui
except ImportError:  # pragma: no cover
    pyqt_found = False
else:
    pyqt_found = True


def plot_transform(transform, scales=None, multiview=False):
    """Display the different bands on the requested scales.

    Parameters
    ----------
    transform: WaveletTransformBase
        a wavelet decomposition.
    scales: list, default None
        the desired scales, if None compute at all scales.
    multiview: bool, default False
        if True use a slider to select a specific band.

    Notes
    -----
    This function is deprecated and will be removed in a future release.

    """
    if not PYQT_FOUND:
        raise ImportError(
            'To use this deprecated function you will need to install '
            + 'pyqtgraph manually. Note that the current implementation '
            + 'is not compatible with PyQT v6.'
        )

    # Set default scales
    scales = scales or range(transform.nb_scale)

    # Create application and tab widget
    app = pyqtgraph.mkQApp()
    tabs = QtGui.QTabWidget()
    tabs.setWindowTitle("Wavelet Transform")

    # Go through each scale
    pen = pyqtgraph.intColor(2)
    for scale in scales:

        # Create the plots for this scales with scrolling possibilities
        scroller = QtGui.QScrollArea()
        tabs.addTab(scroller, "Scale {0}".format(scale))

        # Go through each band of the current scale
        # > using multiview
        # TODO: update this code
        if multiview:
            raise NotImplementedError(
                "Multiview transform view not yet implemented.")
            window = pyqtgraph.image(numpy.asarray(transform[scale]))
            scroller.setWidget(window)
        # > using mosaic
        else:
            window = pyqtgraph.GraphicsWindow()
            scroller.setWidget(window)

            scale_data = transform[scale]
            if not isinstance(scale_data, list):
                scale_data = [scale_data]

            for index, subband_data in enumerate(scale_data):

                # Deal with complex data
                if numpy.iscomplex(subband_data).any():
                    subband_data = numpy.abs(subband_data)
                subband_data = numpy.lib.pad(
                    subband_data, 1, "constant",
                    constant_values=subband_data.max())

                # Select viewer
                if subband_data.ndim == 1:
                    ax = window.addPlot()
                    ax.plot(subband_data, pen=pen)
                elif subband_data.ndim == 2:
                    row = index // 2
                    col = index % 2
                    box = window.addViewBox(row=row, col=col, border="00ff00",
                                            lockAspect=True, enableMouse=False)
                    image = pyqtgraph.ImageItem(subband_data)
                    box.addItem(image)
                else:
                    raise ValueError("This function currently support only "
                                     "1D or 2D data.")
                window.nextRow()

    # Display the tab
    tabs.show()

    # Run application
    app.exec_()
