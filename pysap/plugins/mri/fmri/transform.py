import pysap
from .wavelet_1d.utils import load_transform as load_t_transform
# from wavelet_1d.utils import flatten as fl
# from wavelet_1d.utils import unflatten as ufl
from pysap.plugins.mri.reconstruct.utils import flatten, unflatten

import numpy as np


class FTransform(object):
    def __init__(self, wavelet_name, nb_scale, wavelet_name_t=None, nb_scale_t=1, verbose=0):
        if wavelet_name not in pysap.AVAILABLE_TRANSFORMS:
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        transform_klass = pysap.load_transform(wavelet_name)
        self.s_transform = transform_klass(nb_scale=nb_scale, verbose=verbose)
        if wavelet_name_t is not None:
            self.t_transform = load_t_transform(wavelet_name_t, nb_scale_t, verbose)
        else:
            if verbose:
                print("Linear operator does not have a temporal dimension")
            self.t_transform = None

    def analysis(self, data):
        if isinstance(data, np.ndarray):
            data = pysap.Image(data=data)
        coeffs = []
        coeffs_shape = None
        for t in range(data.shape[-1]):
            self.s_transform.data = data[:, :, t]
            self.s_transform.analysis()
            coeffs_, coeffs_shape = flatten(self.s_transform.analysis_data)
            coeffs.append(coeffs_)
        coeffs = np.asarray(coeffs)
        print(coeffs.shape)
        if self.t_transform is not None:
            coeffs_t = []
            coeffs_t_shape = None
            n = len(coeffs[1])
            for i in range(n):
                self.t_transform.data = coeffs[:, i]
                self.t_transform.analysis()
                coeffs_t_ = self.t_transform.analysis_data
                # coeffs_t_, coeffs_t_shape = fl(self.t_transform.analysis_data)
                coeffs_t.append(coeffs_t_)
        else:
            coeffs_t = coeffs
            coeffs_t_shape = coeffs_shape
        return coeffs_t, coeffs_t_shape

