##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Wavelet transform module.
"""

# System import
from __future__ import division, print_function, absolute_import
from numbers import Number
import os
import numpy
import tempfile

# Package import
import pisap
from pisap.base.exceptions import Exception
from pisap.base.nodes import Node2D
from pisap.base.nodes import Node1D
from .wavelet import Wavelet
from pisap.plotting import plot_transform


def WaveletTransform(data, wavelet, maxscale, mode="symmetric", use_isap=False,
                     **kwargs):
    """ Create a factory to select the proper node type automatically.
    Currently support only 1D or 2D data.

    Parameters
    ----------
    data: nd-array
        input data/signal.
    wavelet: str
        wavelet used in Discrete Wavelet Transform (DWT) decomposition
        and reconstruction.
    maxscale: int
        the maximum scale of decomposition.
    mode: str (optional, default 'symmetric')
        signal extension mode for the 'dwt' and 'idwt' decomposition and
        reconstruction functions.
    use_isap: bool (optional, default False)
        if True, use the C++ optimized ISAP library for the wavelet
        decomposition/reconstruction.
    kwargs: dict
        the parameters used to construct the wavlet.

    Returns
    -------
    transform: WaveletTransformBase
        the desired transform.
    """
    # Check data dimension
    if data is not None:
        ndim = data.ndim
        if ndim not in [1, 2]:
            raise Exception("The WaveletTransform currently support only 1D "
                            "or 2D signals.")
    else:
        ndim = 0

    # Create object
    if ndim in (0, 1):
        cls = type("WaveletTransform1D", (WaveletTransformBase, Node1D),
                   {"x": "a", "y": "d"})
    else:
        cls = type("WaveletTransform2D", (WaveletTransformBase, Node2D),
                   {"x": "l", "y": "h"})
    return cls(data, wavelet, maxscale, mode, use_isap, **kwargs)


class WaveletTransformBase(object):
    """ Data structure representing wavelet decomposition of a signal.
    """
    # Define class parameters
    x = None
    y = None

    def __init__(self, data, wavelet, maxscale, mode="symmetric",
                 use_isap=False, **kwargs):
        """ Initialize the WaveletTransform class.

        Parameters
        ----------
        data: nd-array
            input data/signal.
        wavelet: str
            wavelet used in Discrete Wavelet Transform (DWT) decomposition
            and reconstruction.
        maxscale: int
            the maximum scale of decomposition.
        mode: str (optional, default 'symmetric')
            signal extension mode for the 'dwt' and 'idwt' decomposition and
            reconstruction functions.
        use_isap: bool (optional, default False)
            if True, use the C++ optimized ISAP library for the wavelet
            decomposition/reconstruction.
        kwargs: dict
            the parameters used to construct the wavlet.
        """
        # Get the maximum scale
        if data is not None:
            self._data = numpy.asarray(data, dtype=numpy.double)
            self.data_size = data.shape
        else:
            self.data_size = None

        # Define class attributes
        self.use_isap = use_isap
        self.mode = mode
        self._maxscale = maxscale
        self.isap_trf_header = None

        # Inheritance
        if self.use_isap:
            super(WaveletTransformBase, self).__init__(
                parent=None, data=None, node_name="")
            self.wavelet = None
            self.walk()     
        else:
            super(WaveletTransformBase, self).__init__(
                parent=None, data=self._data, node_name="")
            self.wavelet = Wavelet(wavelet, **kwargs)

    def get_scale(self, scale, order="natural", decompose=True):
        """ Returns all nodes from specified scale.

        Parameters
        ----------
        scale: int
            decomposition scale from which the nodes will be collected.
        order: str (optional, default 'natural')
            if 'natural' a flat list is returned,
            else if 'freq' a 2d structure with rows and cols sorted by
            corresponding dimension frequency of 2d coefficient array
            (adapted from 1d case).
        decompose: bool (optional, default True)
            if set then the method will try to decompose the data up
            to the specified scale.

        Returns
        -------
        nodes: list or list of list
            the nodes with requested scale and ordering.
        """
        # Check input parameters
        if order not in ("natural", "freq"):
            raise ValueError("Unrecognize order '{0}'.".format(order))
        if scale > self.maxscale:
            raise ValueError(
                "The desired scale '{0}' cannot be greater than the "
                "maximum decomposition scale value '{1}'.".format(
                    scale, self.maxscale))

        # Get the node of desired scale
        result = []
        def collect(node):
            if node.scale == scale:
                result.append(node)
                return False
            return True
        self.walk(collect, decompose=decompose)

        # Sort the nodes based on the requested order
        if order == "freq":
            nodes = {}
            for (row_path, col_path), node in [
                    (self.expand_path(node.path), node) for node in result]:
                nodes.setdefault(row_path, {})[col_path] = node
            graycode_order = get_graycode_order(scale, x=self.x, y=self.y)
            nodes = [nodes[path] for path in graycode_order if path in nodes]
            result = []
            for row in nodes:
                result.append(
                    [row[path] for path in graycode_order if path in row])

        return result

    def show(self, scales=None):
        """ Display the different bands on the requested scales.

        Parameters
        ----------
        scales: list  (optional, deafault None)
            the desired scales, if None compute at all scales.
        """
        if self.use_isap:
            cube = pisap.Image(data=self.to_cube()[:, 0],
                               metadata=self.isap_trf_header)
            cube.show()
        else:
            scales = scales or range(self.maxscale)
            plot_transform(self, scales)

    def set_constant_values(self, values):
        """ Set constant values on each scale.

        Parameters
        ----------
        values: list of float or float
            the values to be associated to each scale.
        """
        if not isinstance(values, list):
            values = [values] * (self._maxscale + 1)
        for scale in range(self._maxscale + 1):
            for node in self.get_scale(scale):
                if node.data is not None:
                    node.data *= 0  
                    node.data += values[scale]

    def analysis(self, **kwargs):
        """ Decompose all the nodes.

        Parameters
        ----------
        kwargs: dict (optional)
            the parameters that will be passed to
            'pisap.extensions.mr_tansform'.
        """
        # Check input parameters
        if "number_of_scales" in kwargs:
            print("The 'number_of_scales' mr_tansform parameter was set "
                  "automatically to '{0}'.".format(self.maxscale + 1))
        kwargs["number_of_scales"] = self.maxscale + 1

        # Create nodes/decompose
        self.walk()

        # Insert the ISAP decomposition result
        if self.use_isap:
            tmpdir = tempfile.mkdtemp()
            in_image = os.path.join(tmpdir, "in.fits")
            out_mr_file = os.path.join(tmpdir, "cube.mr")
            try:
                pisap.io.save(self._data, in_image)
                pisap.extensions.mr_transform(in_image, out_mr_file, **kwargs)
                image = pisap.io.load(out_mr_file)
                cube = image.data
                self.isap_trf_header = image.metadata
            except:
                raise
            finally:
                for path in (in_image, out_mr_file):
                    if os.path.isfile(path):
                        os.remove(path)
                os.rmdir(tmpdir)
            self.from_cube(cube)

    def synthesis(self):
        """ Reconstruct the image from all nodes.

        Returns
        -------
        image: pisap.Image
            the reconsructed image.
        """
        # Initilize the output image
        image = None

        # Use ISAP for the reconstruction
        if self.use_isap:
            cube = pisap.Image(data=self.to_cube()[:, 0],
                               metadata=self.isap_trf_header)
            tmpdir = tempfile.mkdtemp()
            in_mr_file = os.path.join(tmpdir, "cube.mr")
            out_image = os.path.join(tmpdir, "out.fits")
            try:
                pisap.io.save(cube, in_mr_file)
                pisap.extensions.mr_recons(in_mr_file, out_image, verbose=True)
                image = pisap.io.load(out_image)
            except:
                raise
            finally:
                for path in (in_mr_file, out_image):
                    if os.path.isfile(path):
                        os.remove(path)
                os.rmdir(tmpdir)

        # Use the embedded python code otherwise
        else:
            image = pisap.Image(data=self.reconstruct())

        return image


def get_graycode_order(scale, x="a", y="d"):
    """ Get the decomposition mosaic.

    Parameters
    ----------
    scale: int
        the decomposition scale.
    x: str (optional, default 'a')
        the x assocaited label.
    y: str (optional, default 'b')
        the x assocaited label.
    
    Returns
    -------
    graycode_order: list of str
        the mosaic.
    """
    graycode_order = [x, y]
    for i in range(scale - 1):
        graycode_order = [x + path for path in graycode_order] +[
                          y + path for path in graycode_order[::-1]]
    return graycode_order


