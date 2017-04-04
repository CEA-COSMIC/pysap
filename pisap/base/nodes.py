##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy

# Package import
from .wavelet_tree import BaseNode
from .ndwt import dwt2, idwt2
from .ndwt import dwt1, idwt1


class Node1D(BaseNode):
    """ WaveletTransform 1D tree node.

    Subnodes are called 'a' and 'd', just like approximation
    and detail coefficients in the Discrete Wavelet Transform.
    """
    A = "a"
    D = "d"
    PARTS = A, D
    PART_LEN = 1

    def _create_subnode(self, part, data=None, overwrite=True):
        """ Method to create a specific subnode.

        Parameters
        ----------
        part: str
            the node name.
        data: ndarray (optional, default None)
            the node attached data.
        overwrite: bool (optional, default True)
            if True, overwrite existing node.
        """
        return self._create_subnode_base(node_cls=Node1D, part=part, data=data,
                                         overwrite=overwrite)

    def _decompose(self):
        """ Decompose a node attached data.

        See also
        --------
        dwt1: for 1D Discrete Wavelet Transform output coefficients.
        """
        # DWT
        # > empty data case
        if self.is_empty:
            data_a, data_d = None, None
            if self._get_node(self.A) is None:
                self._create_subnode(self.A, data_a)
            if self._get_node(self.D) is None:
                self._create_subnode(self.D, data_d)
        # > with data case
        else:
            data_a, data_d = dwt1(self.data, self.wavelet, self.mode)
            self._create_subnode(self.A, data_a)
            self._create_subnode(self.D, data_d)

        return self._get_node(self.A), self._get_node(self.D)

    def _reconstruct(self, update):
        """ Reconstruct the node attached data.

        Parameters
        ----------
        update: bool
            if True, then reconstructed data replaces the current node data.
        """
        # Get the node attached data
        data_a, data_d = None, None
        node_a, node_d = self._get_node(self.A), self._get_node(self.D)
        if node_a is not None:
            data_a = node_a.reconstruct()
        if node_d is not None:
            data_d = node_d.reconstruct()

        # iDWT
        # > missing data
        if data_a is None and data_d is None:
            raise ValueError(
                "Tree is missing data - all subnodes of '{0}' node "
                "are None. Cannot reconstruct node.".format(self.path))
        # > with data
        else:
            rec = idwt1(data_a, data_d, self.wavelet, self.mode)
            if update:
                self.data = rec
            return rec

    def expand_path(self, path):
        """ Transcoding table.
        """
        expanded_paths = {
            self.A: "a",
            self.D: "d",
        }
        return ("".join([expanded_paths[p] for p in path]),
                "".join([expanded_paths[p] for p in path]))

    def to_cube(self):
        """ Convert the tree object to a cube ie an array.

        Returns
        -------
        cube: ndarray
            the generated cube.
        """
        if self.parent is None:
            cubes = numpy.zeros(
                (self._maxscale, len(self.PARTS)) + self.data_size)
            for scale in range(self._maxscale):
                for band, node in enumerate(self.get_scale(scale)):
                    cubes[scale, band, :node.data.shape[0]] = node.data
        else:
            cubes = None
        return cubes


class Node2D(BaseNode):
    """ WaveletTransform 2D tree node.

    Subnodes are called 'a' (LL), 'h' (HL), 'v' (LH) and  'd' (HH), like
    approximation and detail coefficients in the 2D Discrete Wavelet Transform
    where:

    .. code::

                                    -----------------
                                    |       |       |
                                    |cA(LL) |cH(HL) |
                                    |       |       |
        (cA, (cH, cV, cD))  <--->   -----------------
                                    |       |       |
                                    |cV(LH) |cD(HH) |
                                    |       |       |
                                    -----------------

    """
    LL = "a"  # Approximation coefficients
    HL = "h"  # Horizontal detail coefficients
    LH = "v"  # Vertical detail coefficients
    HH = "d"  # Diagonal detail coefficients

    PARTS = LL, HL, LH, HH
    PART_LEN = 1

    def _create_subnode(self, part, data=None, overwrite=True):
        """ Method to create a specific subnode.

        Parameters
        ----------
        part: str
            the node name.
        data: ndarray (optional, default None)
            the node attached data.
        overwrite: bool (optional, default True)
            if True, overwrite existing node.
        """
        return self._create_subnode_base(node_cls=Node2D, part=part, data=data,
                                         overwrite=overwrite)

    def _decompose(self):
        """ Decompose the node attached data.

        See also
        --------
        dwt2: for 2D Discrete Wavelet Transform output coefficients.
        """
        # DWT
        # > empty data case
        if self.is_empty:
            data_ll, data_lh, data_hl, data_hh = None, None, None, None
        # > with data case
        else:
            data_ll, (data_hl, data_lh, data_hh) = dwt2(
                self.data, self.wavelet, self.mode)

        # Create wavelet tree nodes
        self._create_subnode(self.LL, data_ll)
        self._create_subnode(self.LH, data_lh)
        self._create_subnode(self.HL, data_hl)
        self._create_subnode(self.HH, data_hh)

        return (self._get_node(self.LL), self._get_node(self.HL),
                self._get_node(self.LH), self._get_node(self.HH))

    def _reconstruct(self, update):
        """ Reconstruct the node attached data.

        Parameters
        ----------
        update: bool
            if True, then reconstructed data replaces the current node data.
        """
        # Get the node attached data
        data_ll, data_lh, data_hl, data_hh = None, None, None, None
        node_ll, node_lh, node_hl, node_hh = (
            self._get_node(self.LL), self._get_node(self.LH),
            self._get_node(self.HL), self._get_node(self.HH))
        if node_ll is not None:
            data_ll = node_ll.reconstruct()
        if node_lh is not None:
            data_lh = node_lh.reconstruct()
        if node_hl is not None:
            data_hl = node_hl.reconstruct()
        if node_hh is not None:
            data_hh = node_hh.reconstruct()

        # iDWT
        # > missing data
        if (data_ll is None and data_lh is None
                and data_hl is None and data_hh is None):
            raise ValueError(
                "Tree is missing data - all subnodes of '{0}' node "
                "are None. Cannot reconstruct node.".format(self.path))
        # > with data
        else:
            coeffs = data_ll, (data_hl, data_lh, data_hh)
            rec = idwt2(coeffs, self.wavelet, self.mode)
            if update:
                self.data = rec
            return rec

    def expand_path(self, path):
        """ Transcoding table.
        """
        expanded_paths = {
            self.HH: "hh",
            self.HL: "hl",
            self.LH: "lh",
            self.LL: "ll"
        }
        return ("".join([expanded_paths[p][0] for p in path]),
                "".join([expanded_paths[p][1] for p in path]))

    def to_cube(self, verbose=0):
        """ Convert the tree object to a cube ie an array.

        Returns
        -------
        cube: ndarray
            the generated cube.
        verbose: int (optional, default 0)
            the verbosity level
        """
        if self.parent is None:
            cube = numpy.zeros(
                (self._maxscale + 1, len(self.PARTS)) + self.data_size)
            cube[0, 0] = self[""].data
            for scale in range(self._maxscale):
                for band, part in enumerate(self.PARTS):
                    node = self["a" * scale + part]
                    if node.data is not None:
                        node_shape = node.data.shape
                        cube[scale + 1, band, :node_shape[0],
                              :node_shape[1]] = node.data
                    elif verbose > 0:
                        print("Empty node '{0}' skiped in 'to_cube' "
                              "method.".format("a" * scale + part))
        else:
            print("Genererate the cube only from the root node.")
            cube = None

        return cube

    def from_cube(self, cube):
        """ Set the tree decomposition coefficients from a cube.

        Parameters
        ----------
        cube: ndarray (nb_scales, X, Y)
            the cube that containes the decomposition coefficients.
        """
        if self.parent is None:
            if cube.ndim != 3:
                raise ValueError(
                    "Expect a cubes of the form (nb_scales, X, Y).")
            part = ""
            for data_scale in cube:
                self[part].data = data_scale
                part += "a"
        else:
            print("Set decomposition coefficients only from the root node.")

