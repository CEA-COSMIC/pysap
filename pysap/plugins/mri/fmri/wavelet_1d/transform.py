import pywt


class OneDWaveletTransformBase:
    def __init__(self, wavelet_name, nb_scale, verbose=0):
        self.nb_scale = nb_scale
        self.name = wavelet_name
        self.filter_bank = None
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
    __use_pywt__ = 1

    def _analysis(self, data):
        coeffs = pywt.wavedec(data, wavelet=self.name, level=self.nb_scale)
        analysis_data, analysis_header = pywt.coeffs_to_array(coeffs)
        return analysis_data, analysis_header

    def _synthesis(self, analysis_data, analysis_header):
        coeffs = pywt.array_to_coeffs(analysis_data, analysis_header, output_format='wavedec')
        data = pywt.waverec(coeffs, wavelet=self.name)
        return data

