# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import pywt


class OneDWaveletTransformBase:
    """
    One dimensional wavelet transform base class
    """
    def __init__(self, wavelet_name, nb_scale):
        """
        Initialization method
        :param wavelet_name: str
            1D Wavelet name
        :param nb_scale: int
            Number of decomposition scales
        """
        self.nb_scale = nb_scale
        self.name = wavelet_name
        self.use_pywt = None

        self._data = None
        self._analysis_data = None
        self._analysis_header = None

    def _set_data(self, data):
        self._data = data
        self._set_transformation_parameters()

    def _get_data(self):
        return self._data

    def _set_analysis_data(self, data):
        self._analysis_data = data

    def _get_analysis_data(self):
        return self._analysis_data

    def _set_analysis_header(self, analysis_header):
        self._analysis_header = analysis_header

    def _get_analysis_header(self):
        return self._analysis_header

    def analysis(self):
        if self._data is None:
            raise ValueError("Please specify first the input data.")
        self._analysis_data, self._analysis_header = self._analysis(self._data)

    def _analysis(self, data):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivative classes.")

    def synthesis(self):
        if self._analysis_data is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients array.")
        if self.use_pywt and self._analysis_header is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients header.")
        return self._synthesis(self._analysis_data, self._analysis_header)

    def _synthesis(self, analysis_data, analysis_header):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivative classes.")

    def _set_transformation_parameters(self):
        self.use_pywt = self.__use_pywt__

    data = property(_get_data, _set_data)
    analysis_data = property(_get_analysis_data, _set_analysis_data)
    analysis_header = property(_get_analysis_header, _set_analysis_header)


class PyWTransform(OneDWaveletTransformBase):
    """
    One dimensional wavelet class using pywt
    """
    __use_pywt__ = 1

    def _analysis(self, data):
        coeffs = pywt.wavedecn(data, wavelet=self.name,
                               level=self.nb_scale, axes=1)
        analysis_data, analysis_header = pywt.coeffs_to_array(coeffs, axes=[1])
        return analysis_data, analysis_header

    def _synthesis(self, analysis_data, analysis_header):
        coeffs = pywt.array_to_coeffs(analysis_data, analysis_header)
        data = pywt.waverecn(coeffs, wavelet=self.name, axes=[1])
        return data

