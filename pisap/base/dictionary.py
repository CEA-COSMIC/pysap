##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module define the base structure that hold all transforms.
"""
import copy
import numpy as np
import os
import re
import tempfile
import matplotlib.pyplot as plt

import pisap
from pisap.base.formating import FLATTENING_FCTS, INFLATING_FCTS
from pisap.base.utils import to_2d_array, stable_mr_recons, stable_mr_transform


class DictionaryBase(object):
    """ DictionaryBase for dictionnary transform tree nodes.

    The DictionaryBase is a base class for all nodes.
    It should not be used directly unless creating a new transformation
    type. It is included here to document the common node interface
    and dictionnary transform classes.
    """

    def __init__(self, data=None, name=None, id_formating=None,
                 bands_names=None, nb_scale=None, nb_band_per_scale=None,
                 bands_lengths=None, bands_shapes=None):
        """ Initialize the DictionaryBase class.

        Parameters
        ----------
        name: str, store the name of the decomposition.

        bands_names: list of str, the name of the different bands.

        id_formating: int, the id of the formating function to use in 'from_cube',
            see pisap/base/formating.py for more details.

        data: np.ndarray, ndim=1, the value of the decomposition on the
            dictionnary.

        nb_scale: int, the number of scale off the decomposition include the
            approximation scale.

        nb_band_per_scale: np.ndarray, ndim=1, dtype=int, vector holding the number of
            band per scale.

        bands_lengths: np.ndarray, ndim=2, (nb_scale, max(nb_band_per_scale)),
            dtype=int, array holding the length between two bands, on the data
            vector, per scale. Useless cell content 0.

        bands_shapes: list of list of tuple, ndim=2,
            (nb_scale, nb_band_per_scale), structure holding the shape of each
            bands, per scale.
        """
        self.name = name
        self.bands_names = bands_names
        self.is_transform = False
        self.isap_trf_header = None
        self.id_formating = id_formating

        self._data = data.flatten()
        self._native_image_shape = data.shape

        self.nb_scale = nb_scale
        self.nb_band_per_scale = nb_band_per_scale

        self.scales_lengths = bands_lengths.sum(axis=1)
        self.scales_padds = np.zeros(bands_lengths.shape[0]+1, dtype=int)
        self.scales_padds[1:] = self.scales_lengths.cumsum()

        self.bands_lengths = bands_lengths
        self.bands_shapes = bands_shapes

        if not self.metadata_is_valid():
            raise ValueError("in {0}: 'metadata' ".format(type(self))
                                + "passed for init is not valid.")

    def check_same_metadata(self, other):
        """ Check if other is a DictionaryBase with the same metadata.

        Parameters
        ----------
        other: DictionaryBase, the object that will be check.

        Returns
        -------
        out: bool, the result, True if same metadata.
        """
        if not isinstance(other, DictionaryBase):
            raise ValueError("'other' should be a {0} ".format(DictionaryBase)
                              + "and not a {0}.".format(type(other)))
        return (self.nb_scale == other.nb_scale and
                (self.nb_band_per_scale == other.nb_band_per_scale).all() and
                (self.bands_lengths == other.bands_lengths).all())

    def metadata_is_valid(self):
        """ Check if the metadata are coherent w.r.t _data.

        Returns
        -------
        out: bool, result, True if it's coherent.
        """
        if self.is_transform:
            return len(self._data) == int(self.bands_lengths.sum())
        else:
            nx, ny = self._native_image_shape
            return len(self._data) == int(nx*ny)

    def __ge__(self, other):
        """ Overload the greater or equal operator. Make a deep copy.

        Parameters
        ----------
        other: DictionaryBase, the object that will be added.

        Returns
        -------
        out: DictionaryBase, the element-wise result (True or False).
        """
        if not self.check_same_metadata(other):
            raise ValueError("Can only check greater or equal DictionaryBase"
                               + "with DictionaryBase.")
        cpy = copy.deepcopy(self)
        cpy._data = self._data >= other._data
        return cpy

    def __add__(self, other):
        """ Overload the add operator. Make a deep copy.

        Parameters
        ----------
        other: DictionaryBase, the object that will be added.

        Returns
        -------
        out: DictionaryBase, the addition result.
        """
        if not self.check_same_metadata(other):
            raise ValueError("Can only add DictionaryBase with DictionaryBase.")
        cpy = copy.deepcopy(self)
        cpy._data = self._data + other._data
        return cpy

    def __sub__(self, other):
        """ Overload the sub operator. Make a deep copy.

        Parameters
        ----------
        other: DictionaryBase, the object that will be subtract.

        Returns
        -------
        out: DictionaryBase, the substraction result.
        """
        if not self.check_same_metadata(other):
            raise ValueError("Can only substract DictionaryBase with DictionaryBase.")
        cpy = copy.deepcopy(self)
        cpy._data = self._data - other._data
        return cpy

    def __mul__(self, coef):
        """ Overload the mul operator.

        Parameters
        ----------
        coef: numeric or list of numerics, the multiplication coefficient.
            If a list is passed, its length should equal to nb_scale for a scale
            specification of coefficients.

        Returns
        -------
        out: DictionaryBase, the multiplied result.
        """
        # list of scalar case
        if isinstance(coef, list):
            if len(coef) != self.nb_scale:
                raise ValueError("Can only multiple list numerics with DictionaryBase "+
                                    "if len(list) == nb_scale.")
            for coef_ in coef:
                if not isinstance(coef_, (int, long, float, complex)):
                    raise ValueError("Can only multiple numerics with DictionaryBase.")
            cpy = copy.deepcopy(self)
            for i, scale in enumerate(cpy): # default iter is band-iteration
                # 'scale' point on self._data[:]
                scale *= coef[i]
        # scalar case
        elif isinstance(coef, (int, long, float, complex)):
            cpy = copy.deepcopy(self)
            cpy._data *= coef
        elif isinstance(coef, DictionaryBase):
            cpy = copy.deepcopy(self)
            cpy._data *= coef._data
        else:
            raise ValueError("Wrong format of 'other' __mul__ or __div__ only "
                       + "accept numerics, list of numerics or DictionaryBase")
        return cpy

    def __div__(self, coef):
        """ Overload the div operator.

        Parameters
        ----------
        coef: numeric or list of numerics, the multiplication coefficient.
            If a list is passed, its length should equal to nb_scale for a scale
            specification of coefficients.

        Returns
        -------
        out: DictionaryBase, the divided result.
        """
        # list of scalar case
        if isinstance(coef, list):
            if len(coef) != self.nb_scale:
                raise ValueError("Can only multiple list numerics with DictionaryBase "+
                                    "if len(list) == nb_scale.")
            for coef_ in coef:
                if not isinstance(coef_, (int, long, float, complex)):
                    raise ValueError("Can only multiple numerics with DictionaryBase.")
            cpy = copy.deepcopy(self)
            for i, scale in enumerate(cpy): # default iter is band-iteration
                # 'scale' point on self._data[:]
                scale /= coef[i]
        # scalar case
        elif isinstance(coef, (int, long, float, complex)):
            cpy = copy.deepcopy(self)
            cpy ._data /= coef
        elif isinstance(coef, DictionaryBase):
            cpy = copy.deepcopy(self)
            cpy._data /= coef._data
        else:
            raise ValueError("Wrong format of 'other' __mul__ or __div__ only "
                       + "accept numerics, list of numerics or DictionaryBase")
        return cpy

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__
    __truediv__ = __div__

    def __repr__(self):
        """ Define the instance string representation.
        """
        return str(self.metadata)

    def __iter__(self):
        """ Define an iterator on the scale.

        Returns
        -------
        scale: np.ndarry, during iteration.
        """
        for i in range(self.nb_scale):
            start_padd = self.scales_padds[i]
            stop_padd = self.scales_padds[i+1]
            yield self._data[start_padd:stop_padd]

    def display_image(self):
        """ Display the specified band.
        """
        if self.is_transform:
            raise ValueError("can't display native image if data transformed")
        img = self._data.reshape(self._native_image_shape)
        plt.imshow(img, cmap=plt.get_cmap('gist_stern'))
        plt.title("Native image")
        plt.show()

    def display_band(self, ks, kb):
        """ Display the specified band.
        """
        if not self.is_transform:
            raise ValueError("can't display band if data not transformed")
        band = self.get_band(ks, kb).reshape(self.bands_shapes[ks][kb])
        plt.imshow(band, cmap=plt.get_cmap('gist_stern'))
        plt.title("scale={0},band={1}".format(ks, kb))
        plt.show()

    def display_all_scales(self):
        """ Display all scales.
        """
        if not self.is_transform:
            raise ValueError("can't display scale if data not transformed")
        max_nb_band = self.nb_band_per_scale.max()
        fig, axes = plt.subplots(self.nb_scale, max_nb_band)
        axes = to_2d_array(axes)
        for ks in range(self.nb_scale):
            for kb in range(self.nb_band_per_scale[ks]):
                axe = axes[ks, kb]
                band = self.get_band(ks, kb).reshape(self.bands_shapes[ks][kb])
                axe.imshow(band, cmap='gist_stern')
                axe.set_title("scale={0} band='{1}'".format(ks, self.bands_names[kb]))
        fig.suptitle("Decomposition '{0}'".format(self.name), fontsize=15)
        plt.show()

    def display_scale(self, ks):
        """ Display the specified scale.
        """
        if not self.is_transform:
            raise ValueError("can't display scale if data not transformed")
        nb_band = self.nb_band_per_scale[ks]
        fig, axes = plt.subplots(1, nb_band)
        axes = [axes] if nb_band == 1 else axes
        for kb, axe in enumerate(axes):
            band = self.get_band(ks, kb).reshape(self.bands_shapes[ks][kb])
            axe.imshow(band, cmap=plt.get_cmap('gist_stern'))
            axe.set_title("band='{1}'".format(ks, self.bands_names[kb]))
        fig.suptitle('#Scale: {0}'.format(ks), fontsize=15)
        plt.show()

    def get_scale(self, ks):
        """ Get the designated scale by its idx_scale.

        Parameters
        ----------
        ks : int, index of the scale.

        Returns
        -------
        band: np.ndarry, the designated band.
        """
        return self._data[self.scales_padds[ks]:self.scales_padds[ks+1]]

    def get_scale_mask(self, ks):
        """ Get a mask for the designated scale by its idx_scale.

        Parameters
        ----------
        ks : int, index of the scale.

        Returns
        -------
        band: np.ndarry, the mask for the designated band.
        """
        mask = np.ndarray((len(self._data),), dtype=bool) * False
        mask[self.scales_padds[ks]:self.scales_padds[ks+1]] = True
        return mask

    def get_band(self, ks, kb):
        """ Get the designated band by its idx_scale and its idx_band.

        Parameters
        ----------
        ks : int, index of the scale.

        kb : int, index of the band.

        Returns
        -------
        band: np.ndarry, the designated band.
        """
        start_scale_padd = self.scales_padds[ks]
        start_band_padd = self.bands_lengths[ks,:kb+1].sum() - self.bands_lengths[ks,kb]
        start_padd = start_scale_padd + start_band_padd
        stop_padd = start_padd + self.bands_lengths[ks, kb]
        band = self._data[start_padd:stop_padd].reshape(self.bands_shapes[ks][kb])
        return band

    def get_band_mask(self, ks, kb):
        """ Get a mask for the designated band by its idx_scale and its idx_band.

        Parameters
        ----------
        ks : int, index of the scale.

        kb : int, index of the band.

        Returns
        -------
        band: np.ndarry, the mask for the designated band.
        """
        mask = np.ndarray((len(self._data),), dtype=bool) * False
        start_scale_padd = self.scales_padds[ks]
        start_band_padd = self.bands_lengths[ks :kb].sum() - self.bands_lengths[ks:0]
        start_padd = start_scale_padd + start_band_padd
        stop_padd = start_padd + self.bands_lengths[ks, kb]
        mask[start_padd:stop_padd] = True
        return mask

    def set_constant_values(self, values):
        """ Set constant values on each scale.

        Parameters
        ----------
        values: list of float or float
            the values to be associated to each scale.
        """
        if not isinstance(values, list):
            values = [values] * self.nb_scale
        for i, scale in enumerate(self): # default iter is band-iteration
            # 'scale' point on self._data[:]
            scale = values[i] * np.ones(len(scale))

    def from_cube(self, cube):
        """ Set the DictionaryBase decomposition coefficients from a cube.

        Parameters
        ----------
        cube: ndarray (nb_scales*bands_per_scale,), the cube that containes
            the decomposition coefficients.
        """
        self._data = FLATTENING_FCTS[self.id_formating](cube, self.nb_scale)
        self.is_transform = True

        if not self.metadata_is_valid():
            raise ValueError("After call 'analysis' 'metadata' is corrupted.")

    def to_cube(self):
        """ Set the DictionaryBase decomposition coefficients from a cube.

        Parameters
        ----------
        cube: ndarray (nb_scales*bands_per_scale,), the cube that containes
            the decomposition coefficients.
        """
        if self.is_empty:
            raise ValueError("call of 'to_cube' with empty _data")
        return INFLATING_FCTS[self.id_formating](self)

    def analysis(self, **kwargs):
        """ Decompose on the dictionnary atoms.

        Parameters
        ----------
        kwargs: dict, the parameters that will be passed to
            'pisap.extensions.mr_tansform'.
        """
        kwargs["number_of_scales"] = self.nb_scale
        kwargs.pop('maxscale')
        tmpdir = tempfile.mkdtemp()
        in_image = os.path.join(tmpdir, "in.fits")
        out_mr_file = os.path.join(tmpdir, "cube.mr")
        try:
            data = self._data.reshape(self._native_image_shape)
            pisap.io.save(data, in_image)
            stable_mr_transform(in_image, out_mr_file, **kwargs)
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
        """ Reconstruct the image the vector data.

        Returns
        -------
        image: pisap.Image
            the reconsructed image.
        """
        cube = pisap.Image(data=self.to_cube(),
                           metadata=self.isap_trf_header)
        tmpdir = tempfile.mkdtemp()
        in_mr_file = os.path.join(tmpdir, "cube.mr")
        out_image = os.path.join(tmpdir, "out.fits")
        try:
            pisap.io.save(cube, in_mr_file)
            stable_mr_recons(in_mr_file, out_image)
            image = pisap.io.load(out_image)
        except:
            raise
        finally:
            for path in (in_mr_file, out_image):
                if os.path.isfile(path):
                    os.remove(path)
            os.rmdir(tmpdir)
        return image

    @property
    def shape(self):
        """ Property to get the shape.
        """
        return self.bands_shapes

    @property
    def cube_shape(self):
        """ Return the shape of the cube in .fits ISAP transform.
        """
        dico_shape = {}
        for key, item in self.isap_trf_header.iteritems():
            if re.match('NAXIS.', key) is not None:
                dico_shape[key] = item
        return tuple([tup[1] for tup in sorted(dico_shape.items(), reverse=True)])

    @property
    def absolute(self):
        """ Define the absolute operator. Make a deep copy.

        Returns
        -------
        out: DictionaryBase, the absolute coefficient on dictionnary.
        """
        cpy = copy.deepcopy(self)
        cpy._data = np.abs(cpy._data)
        return cpy

    @property
    def sign(self):
        """ Define the sign operator. Make a deep copy.

        Returns
        -------
        out: DictionaryBase, the 'sign' coefficient on dictionnary
            (element-wise result).
        """
        cpy = copy.deepcopy(self)
        cpy._data = np.sign(cpy._data)
        return cpy

    @property
    def is_empty(self):
        """ Property to check if the node attached data is empty.

        Returns
        -------
        out: bool, result, True if there is no data.
        """
        return (self._data is None) or (len(self._data) == 0)

    @property
    def metadata(self):
        """ Property to get the metadata concatenate in a Python dictionnary.

        Returns
        -------
        data: metadata ,dictionnary, the metadata.
        """
        return {'name': self.name,
                'bands names': self.bands_names,
                'id_formating': self.id_formating,
                'native_image_shape': self._native_image_shape,
                'nb_scale': self.nb_scale,
                'nb_band_per_scale': self.nb_band_per_scale,
                'bands_shapes': self.bands_shapes,
                }


class Dictionary(object):
    """ Dictionary class.

    This abstract class defines a generic container to define a decomposition
    on the atoms of this dictionnary.
    """

    def __init__(self, **kwargs):
        """ Initialize the Dictionary class.
        """
        self.metadata = {'nb_scale': kwargs['maxscale']}
        self.isap_kwargs = kwargs

    def _late_init(self):
        """ Fill the metadata and isap_kwargs attribute based on the specificity
        of the transform. This function should call _compute_metadata.
        """
        if self.data is None:
            raise("'_late_init' should only be call "
                    + "in 'op' after self.data is set")
        name, bands_names, nb_band_per_scale, bands_lengths, \
              bands_shapes, id_trf, id_formating = self._trf_id()
        self.metadata['name'] = name
        self.metadata['bands_names'] = bands_names
        self.metadata['nb_band_per_scale'] = nb_band_per_scale
        self.metadata['bands_lengths'] = bands_lengths
        self.metadata['bands_shapes'] = bands_shapes
        self.metadata['id_formating'] = id_formating
        self.isap_kwargs["type_of_multiresolution_transform"] = id_trf
        self.isap_kwargs["write_all_bands"] = False
        self.isap_kwargs["write_all_bands_with_block_interp"] = False


    def _trf_id(self):
        """ Return the specificity of the transform.
        This function return 'name', 'bands_names', 'nb_band_per_scale',
        'bands_paddings', 'bands_shapes' and 'id_trf' and 'id_formating'.
        """
        raise NotImplementedError("Abstract class 'Dictionary' should not "
                                    + "instanciate call directly")

    def op(self, data):
        """ Operator.

        This method returns the input data convolved with the dictionnary filter.

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image.

        Returns
        -------
        np.ndarray wavelet convolved data.
        """
        self.data = data
        self._late_init()
        trf = DictionaryBase(data=data, **self.metadata)
        trf.analysis(**self.isap_kwargs)
        return trf

    def adj_op(self, trf, dtype="array"):
        """ Adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        trf : DictionaryBase
            dictionnary coefficients store in a numpy.ndarray.
        dtype: str (optional, default 'array')
            if 'array' return the data as a ndarray, otherwise return an image.

        Returns
        -------
        np.ndarray reconstructed data.
        """
        image = trf.synthesis()
        return image.data if dtype == "array" else image

    def l2norm(self, data_shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        data_shape: uplet
            the ndarray data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data.
        data_shape = np.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = np.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        data = self.op(fake_data)._data

        # Compute the L2 norm
        return np.linalg.norm(data)
