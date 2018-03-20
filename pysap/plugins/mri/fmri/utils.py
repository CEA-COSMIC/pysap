import numpy as np
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations as c_m_l


def convert_mask_to_locations(mask):
    """
    Converts a mask of booleans into a list of samples
    :param mask: np.ndarray
         The mask for the Fourier points.
         If ony one mask is given, it is repeated through the sequence
    :return:
        samples_locations: np.ndarray
            list of the samples between [-0.5, 0.5[.
    """
    if len(mask.shape) == 2 and mask.shape[0] == mask.shape[1]:  # Just one frame
        return c_m_l(mask)
    else:
        loc = []
        for i in range(mask.shape[2]):
            row_, col_ = np.where(mask[:, :, i] == 1)
            row_ = row_.astype("float") / mask.shape[0] - 0.5
            col_ = col_.astype("float") / mask.shape[1] - 0.5
            loc.append(np.c_[row_, col_])
        return np.moveaxis(np.asarray(loc), 0, -1)


def convert_locations_to_mask(samples_locations, img_shape):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters
    ----------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    img_shape: tuple of int
        shape of the desired mask, not necessarily a square matrix.

    Returns
    -------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarily a square matrix.
    """
    samples_locations = samples_locations.astype("float")
    samples_locations += 0.5
    samples_locations[:, 0] *= img_shape[0]
    samples_locations[:, 1] *= img_shape[1]
    samples_locations = np.round(samples_locations)
    samples_locations = samples_locations.astype("int")
    mask = np.zeros(img_shape)
    if len(img_shape) == 2:
        mask[samples_locations[:, 0], samples_locations[:, 1]] = 1
    else:
        for i in range(img_shape[2]):
            mask[samples_locations[:, 0, i], samples_locations[:, 1, i], i] = 1
    return mask
