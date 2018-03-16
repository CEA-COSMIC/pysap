from .transform import PyWTransform


def load_transform(wavelet_name, nb_scale):
    """
    Load a transform using its name
    :param wavelet_name: str
        Name of the 1D wavelet
    :param nb_scale: int
        Number of decomposition scales
    :return: transform
        PyWTransform instance

    """
    return PyWTransform(wavelet_name, nb_scale)
