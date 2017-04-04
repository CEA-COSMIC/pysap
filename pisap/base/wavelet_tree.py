##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import copy
import numpy as np


class BaseNode(object):
    """ BaseNode for wavelet transform tree nodes.

    The BaseNode is a base class for all nodes.
    It should not be used directly unless creating a new transformation
    type. It is included here to document the common node interface 
    and wavelet transform classes.
    """
    # PART_LEN and PARTS attributes that define path tokens for node[] lookup
    # must be defined in subclasses.
    PART_LEN = None
    PARTS = None

    def __init__(self, parent, data, node_name):
        """ Initilaize the BaseNode class.

        Parameters
        ----------
        parent: Node
            parent node - if parent is None then the node is considered detached
            (ie root).
        data: ndarray
            data associated with the node.
        node_name :
            a name identifying the coefficients type.
        """
        # Define class parameters
        self.parent = parent
        if parent is not None:
            self.wavelet = parent.wavelet
            self.mode = parent.mode
            self.scale = parent.scale + 1
            self._maxscale = parent.maxscale
            self.path = parent.path + node_name
        else:
            self.wavelet = None
            self.path = node_name
            self.scale = 0
            if not hasattr(self, "_maxscale"):
                raise ValueError("The 'BaseNode' derived class must define a "
                                 "'_maxscale' parameter.")

        # Data = signal on scale 0, coeffs on higher scales
        self.data = data

        # Define all the nodes parameters
        self._init_subnodes()

    def __add__(self, other):
        """ Overload the add operator.

        Parameters
        ----------
        other: BaseNode or numeric
            the object that will be added.

        Returns
        -------
        out: BaseNode
            the addition result as a tree.
        """
        current_node = copy.deepcopy(self)
        if isinstance(other, BaseNode):
            def func(node1, other):
                other_data = other[node1.path].data
                if node1.data is not None and other_data is not None:
                    node1.data += other_data
                elif node1.data is None and other_data is None: 
                    pass
                else:
                    raise ValueError("Can't add data with None")
        else:
            def func(node1, other):
                if node1.data is not None:
                    node1.data += other
        current_node.walk_depth(func, args=(), kwargs={"other": other},
                                decompose=True)
        return current_node
            

    def __sub__(self, other):
        """ Overload the sub operator.

        Parameters
        ----------
        other: BaseNode or numeric
            the object that will be added.

        Returns
        -------
        out: BaseNode
            the substraction result as a tree.
        """
        current_node = copy.deepcopy(self)
        if isinstance(other, BaseNode):
            def func(node1, other):
                other_data = other[node1.path].data
                if node1.data is not None and other_data is not None:
                    node1.data -= other_data
                elif node1.data is None and other_data is None: 
                    pass
                else:
                    raise ValueError("Can't add data with None")
        else:
            def func(node1, other):
                if node1.data is not None:
                    node1.data -= other
        current_node.walk_depth(func, args=(), kwargs={"other": other},
                                decompose=True)
        return current_node

    def __mul__(self, other):
        """ Overload the mul operator.

        Parameters
        ----------
        other: BaseNode or numeric
            the object that will be added.

        Returns
        -------
        out: BaseNode
            the multiplication result as a tree.
        """
        current_node = copy.deepcopy(self)
        if isinstance(other, BaseNode):
            def func(node1, other):
                other_data = other[node1.path].data
                if node1.data is not None and other_data is not None:
                    node1.data *= other_data
                elif node1.data is None and other_data is None: 
                    pass
                else:
                    raise ValueError("Can't add data with None")
        elif isinstance(other, list):
            def func(node1, other):
                if node1.data is not None:                   
                    node1.data *= other[len(node1.path)]         
        else:
            def func(node1, other):
                if node1.data is not None:
                    node1.data *= other
        current_node.walk_depth(func, args=(), kwargs={"other": other},
                                decompose=True)
        return current_node

    def __div__(self, other):
        """ Overload the div operator.

        Parameters
        ----------
        other: BaseNode or numeric
            the object that will be added.

        Returns
        -------
        out: BaseNode
            the division result as a tree.
        """
        current_node = copy.deepcopy(self)
        if isinstance(other, BaseNode):
            def func(node1, other):
                other_data = other[node1.path].data
                if node1.data is not None and other_data is not None:
                    node1.data /= other_data
                elif node1.data is None and other_data is None: 
                    pass
                else:
                    raise ValueError("Can't add data with None")
        else:
            def func(node1, other):
                if node1.data is not None:
                    node1.data /= other
        current_node.walk_depth(func, args=(), kwargs={"other": other},
                                decompose=True)
        return current_node

    def __ge__(self, other):
        """ Overload the ge operator.

        Parameters
        ----------
        other: BaseNode or numeric
            the object that will be added.

        Returns
        -------
        out: BaseNode
            the test result as a tree.
        """
        current_node = copy.deepcopy(self)
        if isinstance(other, BaseNode):
            def func(node1, other):
                other_data = other[node1.path].data
                if node1.data is not None and other_data is not None:
                    node1.data = node1.data >= other_data
                elif node1.data is None and other_data is None: 
                    pass
                else:
                    raise ValueError("Can't add data with None")
        else:
            def func(node1, other):
                if node1.data is not None:
                    node1.data = node1.data >= other
        current_node.walk_depth(func, args=(), kwargs={"other": other},
                                decompose=True)
        return current_node

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__
    __truediv__ = __div__

    def __sign__(self):
        """ Define the sign operator.

        Returns
        -------
        out: BaseNode
            the nodes sign as a tree.
        """
        current_node = copy.deepcopy(self)
        def func(node):
            if node.data is not None:
                node.data = np.sign(node.data)
        current_node.walk_depth(func, decompose=True)
        return current_node

    def __absolute__(self):
        """ Define the absolute operator.

        Returns
        -------
        out: BaseNode
            the nodes absolute data values as a tree.
        """
        current_node = copy.deepcopy(self)
        def func(node):
            if node.data is not None:
                node.data = np.abs(node.data)
        current_node.walk_depth(func, decompose=True)
        return current_node

    sign = property(__sign__)
    absolute = property(__absolute__)

    def __repr__(self):
        """ Define the instance string representation.
        """
        shape = None
        if hasattr(self.data, "shape"):
            shape = self.data.shape
        return "{0}: {1} - {2} {3}".format(
            super(BaseNode, self).__repr__(), self._evaluate_maxscale(),
            shape, self.path)

    def __getitem__(self, path):
        """ Find node represented by the given path.
        Can access nodes on any scale in the decomposition tree.

        Parameters
        ----------
        path: str
            the desired node name.

        Notes
        -----
        If node does not exist yet, it will be created by decomposition of its
        parent node.
        """
        if isinstance(path, str):
            if (self.maxscale is not None
                    and len(path) > self.maxscale * self.PART_LEN):
                raise IndexError("Path length is out of range.")
            if path:
                return self.get_subnode(path[0: self.PART_LEN], decompose=True)[
                    path[self.PART_LEN:]]
            else:
                return self
        else:
            raise TypeError("Invalid path parameter type - expected string but"
                            " got '{0}'.".format(type(path)))

    def __setitem__(self, path, data):
        """ Set node or node's data in the decomposition tree. Nodes are
        identified by string 'path'.

        Parameters
        ----------
        path: str
            node name.
        data: array or BaseNode subclass
            node associated data.
        """
        if isinstance(path, str):
            if (self.maxscale is not None and
                len(self.path) + len(path) > self.maxscale * self.PART_LEN):
                raise IndexError("Path length out of range.")
            if path:
                subnode = self.get_subnode(path[0: self.PART_LEN], decompose=False)
                if subnode is None:
                    self._create_subnode(path[0: self.PART_LEN], data=None)
                    subnode = self.get_subnode(path[0: self.PART_LEN], decompose=False)
                subnode[path[self.PART_LEN:]] = data
            else:
                if isinstance(data, BaseNode):
                    self.data = numpy.asarray(data.data, dtype=numpy.double)
                else:
                    self.data = numpy.asarray(data, dtype=numpy.double)
        else:
            raise TypeError("Invalid path parameter type - expected string but"
                            " got '{0}'.".format(type(path)))

    def __delitem__(self, path):
        """ Remove node from the tree.

        Parameters
        ----------
        path: str
            node name.
        """
        node = self[path]
        # don't clear node value and subnodes (node may still exist outside
        # the tree)
        parent = node.parent
        node.parent = None  # TODO
        if parent and node.node_name:
            parent._delete_node(node.node_name)

    def get_subnode(self, part, decompose=True):
        """ Returns a subnode or None.

        Parameters
        ----------
        part: str
            subnode name.
        decompose: bool (optional, default True)
            if True and corresponding subnode does not exist, the subnode
            will be created using coefficients from the DWT decomposition of
            the current node.
        """
        self._validate_node_name(part)
        subnode = self._get_node(part)
        if subnode is None and decompose:
            self.decompose()
            subnode = self._get_node(part)
        return subnode

    def walk(self, func=None, args=(), kwargs=None, decompose=True):
        """ Traverses the decomposition tree and calls
        'func(node, *args, **kwargs)' on every node. If 'func' returns True,
        descending to subnodes will continue.

        Parameters
        ----------
        func: callable (optional, default None)
            callable accepting `BaseNode` as the first param and
            optional positional and keyword arguments.
            If None and decompose, generate all the decomposition nodes.
        args: uplet (optional, default ())
            function parameters.
        kwargs: dict (optional, default None)
            function keyword parameters.
        decompose: bool (optional, default True)
            if True, the method will also try to decompose the tree
            up to the maximum scale.
        """
        if func is None:
            func = lambda node: True
        if kwargs is None:
            kwargs = {}
        if func(self, *args, **kwargs) and self.scale < self.maxscale:
            for part in self.PARTS:
                subnode = self.get_subnode(part, decompose)
                if subnode is not None:
                    subnode.walk(func, args, kwargs, decompose)

    def walk_depth(self, func, args=(), kwargs=None, decompose=True):
        """ Walk tree and call func on every node starting from the bottom-most
        nodes.

        Parameters
        ----------
        func: callable
            callable accepting 'BaseNode' as the first param and
            optional positional and keyword arguments
        args: uplet (optional, default ())
            function parameters.
        kwargs: dict (optional, default None)
            function keyword parameters.
        decompose: bool (optional, default True)
            if True, the method will also try to decompose the tree
            up to the maximum scale.
        """
        if kwargs is None:
            kwargs = {}
        if self.scale < self.maxscale:
            for part in self.PARTS:
                subnode = self.get_subnode(part, decompose)
                if subnode is not None:
                    subnode.walk_depth(func, args, kwargs, decompose)
        func(self, *args, **kwargs)

    def get_leaf_nodes(self, decompose=False):
        """ Returns leaf nodes.

        Parameters
        ----------
        decompose: bool (optional, default False)
            if True and corresponding subnode does not exist, the subnode
            will be created using coefficients from the DWT decomposition of
            the current node.
        """
        result = []
        def collect(node):
            if node.scale == node.maxscale and not node.is_empty:
                result.append(node)
                return False
            if not decompose and not node.has_any_subnode:
                result.append(node)
                return False
            return True
        self.walk(collect, decompose=decompose)
        return result

    def decompose(self):
        """ Decompose node data creating DWT coefficients subnodes.
        Performs Discrete Wavelet Transform on the node data.

        Returns
        -------
        coeffs: array
            the transform coefficients.

        Note
        ----
        Descends to subnodes and recursively calls the 'reconstruct'method
        on them.
        """
        if self.scale < self.maxscale:
            return self._decompose()
        else:
            raise ValueError("Maximum decomposition scale reached: "
                             "{0} < {1}.".format(self.scale, self.maxscale))

    def _decompose(self):
        """ Virtual method to decompose a node attached data that must be
        defined in subclasses.
        """
        raise NotImplementedError(
            "The '_decompose' method must be implemented in subclasses.")

    def reconstruct(self, update=False):
        """ Reconstruct node from subnodes.

        Parameters
        ----------
        update: bool (optional, default False)
            if set then the reconstructed data replaces the current node data.

        Returns
        -------
        data: ndarray
            original node data if subnodes do not exist, IDWT of subnodes
            otherwise.
        """
        if not self.has_any_subnode:
            return self.data
        return self._reconstruct(update)

    def _reconstruct(self, update):
        """ Virtual method to recompose a node attached data that must be
        defined in subclasses.
        """
        raise NotImplementedError(
            "The '_reconstruct' method must be implemented in subclasses.")

    ######################################################################
    # Private interface 
    ######################################################################

    def _init_subnodes(self):
        """ Create all the subnodes empty.
        """
        for part in self.PARTS:
            self._set_node(part, None)

    def _create_subnode(self, part, data=None, overwrite=True):
        """ Virtual method to create a node that must be defined in subclasses.
        """
        raise NotImplementedError(
            "The '_create_subnode' method must be implemented in subclasses.")

    def _create_subnode_base(self, node_cls, part, data=None, overwrite=True):
        """ Method to create a specific node.

        Parameters
        ----------
        node_cls: Node
            a specific node
        part: str
            the node name.
        data: ndarray (optional, default None)
            the node attached data.
        overwrite: bool (optional, default True)
            if True, overwrite existing node.
        """
        self._validate_node_name(part)
        if not overwrite and self._get_node(part) is not None:
            return self._get_node(part)
        node = node_cls(self, data, part)
        self._set_node(part, node)
        return node

    def _get_node(self, part):
        """ Get a specific node.
        """
        return getattr(self, part)

    def _set_node(self, part, node):
        """ Add a node as a class parameter.
        """
        setattr(self, part, node)

    def _delete_node(self, part):
        """ Reset a specific node to empty.
        """
        self._set_node(part, None)

    def _validate_node_name(self, part):
        """ Check the node name.
        
        Prameters
        ---------
        part: str
            the node name.
        """
        if part not in self.PARTS:
            raise ValueError(
                "Subnode name must be in [{0}], not '{1}'." .format(
                    self.PARTS, part))

    def _evaluate_maxscale(self):
        """ Try to find the value of maximum decomposition scale, it must be
        specified explicitly.
        """
        if self._maxscale is not None:
            return self._maxscale
        if self.parent is not None:
            return self.parent._evaluate_maxscale(evaluate_from)

        return None

    ######################################################################
    # Properties 
    ######################################################################

    def maxscale(self):
        """ Property to find the value of maximum decomposition scale.

        Returns
        -------
        maxscale: int
            the value of maximum decomposition scale
        """
        self._maxscale = self._evaluate_maxscale()
        return self._maxscale

    def node_name(self):
        """ Property to get the node name.
        """
        return self.path[-self.PART_LEN:]

    def is_empty(self):
        """ Property to check if the node attached data is empty.
        """
        return self.data is None

    def has_any_subnode(self):
        """ Property to check if at least one subnode is crearted.
        """
        for part in self.PARTS:
            if self._get_node(part) is not None:
                return True
        return False

    def shape(self):
        """ Property to get the shape.
        """
        if self.PARTS is None or self.data is None:
            return ()
        return (self.maxscale, len(self.PARTS)) + self.data.shape

    maxscale = property(maxscale)
    node_name = property(node_name)
    is_empty = property(is_empty)
    has_any_subnode = property(has_any_subnode)
    shape = property(shape)
