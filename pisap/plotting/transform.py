##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui
import pyqtgraph


def plot_transform(wavelet, scales):
    """ Display the different bands on the requested scales.

    Parameters
    ----------
    wavelet: Wavelet
        a wavelet.
    scales: list  (optional, deafault None)
        the desired scales, if None compute at all scales.
    """
    # Set default scales
    scales = scales or [4]

    # Create application and tab widget
    app = pyqtgraph.mkQApp()
    tabs = QtGui.QTabWidget()
    tabs.setWindowTitle("Wavelet Transform")

    # Go through each scale
    pen = pyqtgraph.intColor(2)
    for scale in scales:

        # Create the plots for this scales with scrolling possibilities
        scroller = QtGui.QScrollArea()
        window = pyqtgraph.GraphicsWindow()
        scroller.setWidget(window)
        tabs.addTab(scroller, "Scale {0}".format(scale))

        # Go through each band of the current scale
        for row_subband in wavelet.get_scale(scale, order="freq"):
            for subband in row_subband:
                if subband.data.ndim == 1:
                   ax = window.addPlot()
                   ax.plot(subband.data, pen=pen)
                elif subband.data.ndim == 2:
                    box = window.addViewBox()
                    box.setAspectLocked(True)
                    image = pyqtgraph.ImageItem(subband.data)
                    box.addItem(image)
                else:
                    raise ValueError("This function currently support only "
                                     "2D or 3D data.")
            window.nextRow()

    # Display the tab
    tabs.show()

    # Run application
    app.exec_()


def plot_wavelet(wavelet, scales=None):
    """ Display approximations of scaling function (phi) and wavelet
    function (psi) on xgrid (x) at given scales.

    Parameters
    ----------
    wavelet: Wavelet
        a wavelet.
    scales: list (optional, deafault None)
        the desired scales, if None compute at scale 4.
    """
    # Set default scales
    scales = scales or [4]

    # Create application and tab widget
    app = pyqtgraph.mkQApp()
    tabs = QtGui.QTabWidget()
    tabs.setWindowTitle("Wavelets")

    # Go through each scale
    pen = pyqtgraph.intColor(2)
    for scale in scales:

        # Create the plots for this scales with scrolling possibilities
        scroller = QtGui.QScrollArea()
        window = pyqtgraph.GraphicsWindow()
        scroller.setWidget(window)
        if wavelet.wtype == "orthogonal":
            phi, psi, x = wavelet.wavefun(scale=scale)
            ax = window.addPlot(title=wavelet.name + " phi")
            ax.plot(phi, pen=pen)
            bx = window.addPlot(title=wavelet.name + " psi")
            bx.plot(psi, pen=pen)
        else:
            phi, psi, phi_r, psi_r, x = wavelet.wavefun(scale=scale)
            ax = window.addPlot(title=wavelet.name + " phi")
            ax.plot(phi, pen=pen)
            bx = window.addPlot(title=wavelet.name + " psi")
            bx.plot(psi, pen=pen)
            ax = window.addPlot(title=wavelet.name + " phi_r")
            ax.plot(phi_r, pen=pen)
            bx = window.addPlot(title=wavelet.name + " psi_r")
            bx.plot(psi_r, pen=pen)
        tabs.addTab(scroller, "Scale {0}".format(scale))

    # Display the tab
    tabs.show()

    # Run application
    app.exec_()
