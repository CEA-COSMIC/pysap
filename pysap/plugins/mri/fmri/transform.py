import pysap
from .wavelet_1d.utils import load_transform as load_transform_t
from pysap.plugins.mri.reconstruct.utils import flatten, unflatten

import numpy as np


class TransformT(object):
    """
    The 2D+T wavelet transform operator.
    It is composed of a 2D wavelet transform performed with sparse2d and a one dimensional
    Wavelet transform performed with pywt
    """
    def __init__(self, wavelet_name, nb_scale, wavelet_name_t=None, nb_scale_t=1, verbose=0):
        if wavelet_name not in pysap.AVAILABLE_TRANSFORMS:
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        transform_klass = pysap.load_transform(wavelet_name)
        self.transform_s = transform_klass(nb_scale=nb_scale, verbose=verbose)
        if wavelet_name_t is not None:
            self.transform_t = load_transform_t(wavelet_name_t, nb_scale_t)
        else:
            if verbose:
                print("Linear operator does not have a temporal dimension")
            self.transform_t = None
        self.coeffs_shape_s = None
        # self.coeffs_shape = None
        self.data_shape = None
        self._data_shape = None

    def analysis(self, data):
        """
        Perform the analysis transform of the wavelet operator.
        :param data: np.ndarray
        The shape of data is to be [n^2, T] where T is the number of frames and n is the one dimensional
        size of each frame
        :return: coeffs_t : np.ndarray
        the decomposition of the list of frames in the Wavelet2D+T domain.
        :return: coeffs_t_shape :
        The shape of the wavelet coefficients
        """
        self.data_shape = data.shape
        self._data_shape = (int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), data.shape[1])
        coeffs = []
        data = np.reshape(data, self._data_shape)
        for t in range(data.shape[-1]):
            self.transform_s.data = data[:, :, t]
            self.transform_s.analysis()
            coeffs_, self.coeffs_shape_s = flatten(self.transform_s.analysis_data)
            coeffs.append(coeffs_)
        coeffs = np.asarray(coeffs).T
        if self.transform_t is not None:
            self.transform_t.data = coeffs
            self.transform_t.analysis()
            coeffs_t = self.transform_t.analysis_data
        else:
            coeffs_t = coeffs
        return coeffs_t, coeffs_t.shape

    def synthesis(self, coeffs):
        """
        Perform the synthesis transform of the wavelet operator
        :param coeffs: np.ndarray
        Coefficients from the Wavelet2D+T decomposition
        :return: data: np.ndarray
        Array of shape np.data_shape which is [n^2, T]
        """
        if self.transform_t is not None:
            self.transform_t.analysis_data = coeffs
            data_t = self.transform_t.synthesis()
        else:
            data_t = coeffs

        data = []
        for t in range(data_t.shape[1]):
            self.transform_s.analysis_data = unflatten(data_t[:, t], self.coeffs_shape_s)
            data_ = self.transform_s.synthesis()
            data.append(data_.data)
        data = np.moveaxis(np.asarray(data), 0, -1)
        data = np.reshape(data, self.data_shape)
        return data

