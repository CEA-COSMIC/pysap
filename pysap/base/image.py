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

# Package import
from .observable import Observable
from pysap.base.exceptions import Exception
from pysap.plotting import plot_data


class Image(Observable):
    """ Class that defines an image.

    An image contains:
        * data: an array of data stored in a numpy.ndarray
        * data_type: whether the data is scalar, vector or matrix.
        * a dictionary of metadata

    If data_type is 'vector' or 'matrix', an array of dimension N will have a
    spacing of size N-1, respectivelly N-2.

    The following event is allowed:
        * modified
    """
    def __init__(self, shape=None, spacing=None, data_type="scalar",
                 metadata=None, **kwargs):
        """ Create an image that can be modified as a numpy arrray.

        Parameters
        ----------
        shape: uplet (optional, default None)
            set this parameter to created an empty image.
        spacing: uplet (optional, default None)
            the image spacing, if not set consider a default isotropic spacing.
        data_type: str (optional, default 'scalar')
            the image data type: 'scalar', 'vector' or 'matrix'.
        metadata: dict (optional, default None)
            some metadata attached to this image.
        kwargs: dict (optional)
            extra arguments may contain the image data as 'data', the empty
            image data filled value as 'value' or any argument of numpy.ndarray
            constructor.
        """
        # Check input parameters
        if data_type not in ["scalar", "vector", "matrix"]:
            raise Exception("Unknown data type '{0}'.".format(data_type))

        # Define class attributes
        self._scroll_axis = 0
        self.data = None
        self.data_type = data_type
        self.metadata = metadata or {}
        self._spacing = None

        # Initialize the Image class
        Observable.__init__(self, ["modified"])

        # Image data initialization
        if "data" in kwargs:
            self.data = numpy.asarray(kwargs["data"])
            del kwargs["data"]
        else:
            if shape is None:
                raise Exception("Wrong shape '{0}'.".format(shape))
            if "value" in kwargs:
                value = kwargs["value"]
                del kwargs["value"]
            else:
                value = None
            self.data = numpy.ndarray(shape, **kwargs)
            if value is not None:
                self.data.fill(value)

        # Image spacing initialization
        if spacing is None:
            self._set_spacing(self._default_spacing())
        else:
            self._set_spacing(spacing)

    def show(self):
        """ Display the image data.
        """
        plot_data(self.data, scroll_axis=self._scroll_axis)

    def modified(self):
        """ Send a modified signal to the observers.
        """
        self.notify_observers("modified")

    def __getitem__(self, where):
        """ Get an items of the image data.
        """
        return self.data[where]

    def __setitem__(self, where, value):
        """ Set an item to the image data.
        """
        self.data[where] = value

    def __array__(self):
        """ Return image data as a numpy array.
        """
        return numpy.asarray(self.data)

    ######################################################################
    # Properties
    ######################################################################

    def _get_spacing(self):
        """ Get the image spacing.
        """
        return self._spacing

    def _set_spacing(self, spacing):
        """ Set the image spacing.

        Parameters
        ----------
        spacing: uplet
            the image spacing.
        """
        self._spacing = numpy.asarray(spacing, dtype=numpy.single)

    def _get_shape(self):
        """ Get the shape of the image.
        This function accounts for non-scalar data, i.e. 'vector' or 'matrix'
        vs 'scalar' data types.
        """
        if self.data_type == "scalar":
            return self.data.shape
        elif self.data_type == "vector":
            return self.data.shape[:-1]
        elif self.data_type == "matrix":
            return self.data.shape[:-2]

    def _get_dtype(self):
        """ Get the image data type.
        """
        return self.data.dtype

    def _get_ndim(self):
        """ Get the image dimension.
        This function accounts for non-scalar data, i.e. 'vector' or 'matrix'
        vs 'scalar' data types.
        """
        if self.data_type == "scalar":
            return self.data.ndim
        elif self.data_type == "vector":
            return self.data.ndim - 1
        elif self.data_type == "matrix":
            return self.data.ndim - 2

    def _get_scroll_axis(self):
        """ Get the scroll axis.

        Returns
        ----------
        scroll_axis: int
            the scroll axis for 3d data.
        """
        return self._scroll_axis

    def _set_scroll_axis(self, scroll_axis):
        """ Modify the scroll axis.

        Parameters
        ----------
        scroll_axis: int
            the scroll axis for 3d data.
        """
        self._scroll_axis = scroll_axis

    scroll_axis = property(_get_scroll_axis, _set_scroll_axis)
    spacing = property(_get_spacing, _set_spacing)
    shape = property(_get_shape)
    dtype = property(_get_dtype)
    ndim = property(_get_ndim)

    ######################################################################
    # Private interface
    ######################################################################

    def _default_spacing(self):
        """ Return the default image spacing.
        """
        dim = self._get_ndim()
        return numpy.ones(dim, dtype=numpy.single)
