##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common tools for MRI image reconstruction.
"""


# System import
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_title('Slice %s / %s' % (self.ind, self.slices))
        self.im.axes.figure.canvas.draw()


def imshow3D(volume, display=False):
    """ Display an 3D volume on the axes, press "p" or "n" to navigate across
    the slices.

    Parameters
    ----------
    volume: ndarray
        the input volume
    """
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap='gray')
    tracker = IndexTracker(ax, volume)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    if display:
        plt.show()


def process_key(event):
    """ Take into account to take the previous or next slice on the volume

    Parameters
    ----------
    event on the keybord
    """
    fig = event.canvas.figure
    ax = fig.axes[0]
    plt.suptitle('Slice: ', ax.val, ' / ', ax.volume.shape[0])
    if event.key == 'p':
        previous_slice(ax)
    elif event.key == 'n':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    """ Take the previous slice  to take the previous or next slice on the
    volume

    Parameters
    ----------
    ax: axes objects
    """
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    """ Take the previous slice  to take the next or next slice on the
    volume

    Parameters
    ----------
    ax: axes objects
    """
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def flatten_swtn(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = []
    shape_dict = []
    for i in range(len(x)):
        dict_lvl = {}
        for key in x[i].keys():
            dict_lvl[key] = x[i][key].shape
            y = np.concatenate((y, x[i][key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_swtn(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    x = []
    offset = 0
    for i in range(len(shape)):
        sublevel = {}
        for key in shape[i].keys():
            start = offset
            stop = offset + np.prod(shape[i][key])
            offset = stop
            sublevel[key] = y[start: stop].reshape(shape[i][key])
        x.append(sublevel)
    return x


def flatten_wave(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """

    # Flatten the dataset
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape_dict = [x[0].shape]
    for x_i in x[1:]:
        dict_lvl = {}
        for key in x_i.keys():
            dict_lvl[key] = x_i[key].shape
            y = np.concatenate((y, x_i[key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_wave(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    start = 0
    stop = np.prod(shape[0])
    x = [y[start:stop].reshape(shape[0])]
    offset = stop
    for shape_i in shape[1:]:
        sublevel = {}
        for key in shape_i.keys():
            start = offset
            stop = offset + np.prod(shape_i[key])
            offset = stop
            sublevel[key] = y[start: stop].reshape(shape_i[key])
        x.append(sublevel)
    return x


def convert_mask_to_locations_3D(mask):
    """ Return the converted Cartesian mask as sampling locations.

    Parameters
    ----------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.

    Returns
    -------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    """
    dim1, dim2, dim3 = np.where(mask == 1)
    dim1 = dim1.astype("float") / mask.shape[0] - 0.5
    dim2 = dim2.astype("float") / mask.shape[1] - 0.5
    dim3 = dim3.astype("float") / mask.shape[2] - 0.5
    return np.c_[dim1, dim2, dim3]


def normalize_samples(samples_locations):
    """Normalize the 3D samples between [-.5; .5[

    Parameters
    ----------
    samples_locations: np.array
        A representation of the 3D locations of the samples
    """
    samples_locations = samples_locations.astype('float')
    samples_locations[:, 0] /= 2 * np.abs(samples_locations[:, 0]).max()
    samples_locations[:, 1] /= 2 * np.abs(samples_locations[:, 1]).max()
    samples_locations[:, 2] /= 2 * np.abs(samples_locations[:, 2]).max()
    while samples_locations.max() == 0.5:
        dim1, dim2 = np.where(samples_locations == 0.5)
        samples_locations[dim1, dim2] = -0.5
    return samples_locations


def convert_locations_to_mask(samples_locations, img_shape):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters
    ----------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    img_shape: tuple of int
        shape of the desired mask, not necessarly a square matrix.

    Returns
    -------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.
    """
    samples_locations = np.copy(samples_locations).astype("float")
    samples_locations += 0.5
    samples_locations[:, 0] *= img_shape[0]
    samples_locations[:, 1] *= img_shape[1]
    samples_locations[:, 2] *= img_shape[2]
    samples_locations = np.round(samples_locations) - 1
    samples_locations = samples_locations.astype("int")
    mask = np.zeros(img_shape)
    mask[samples_locations[:, 0],
         samples_locations[:, 1],
         samples_locations[:, 2]] = 1
    return mask


def gridding_3d(points, values, img_shape, method='linear'):
    """
    Interpolate non-Cartesian data into a cartesian grid
    Parameters:
    -----------
    points: np.ndarray
        The 3D k_space locations of size [M, 3]
    values: np.ndarray
        The input data value in the points location
    img_shape: tuple
        The final output ndarray shape [N_x, N_y, N_z]
    method: {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation
    Return:
    -------
    np.ndarray
        The gridded solution of shape [N_x, N_y, N_z]
    """
    xi = np.linspace(np.min(points),
                     np.max(points),
                     img_shape[0],
                     endpoint=False)
    yi = np.linspace(np.min(points),
                     np.max(points),
                     img_shape[1],
                     endpoint=False)
    zi = np.linspace(np.min(points),
                     np.max(points),
                     img_shape[1],
                     endpoint=False)
    grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi)
    return griddata(points,
                    values,
                    (grid_x, grid_y, grid_z),
                    method=method,
                    fill_value=0)


def prod_over_maps(S, X):
    """
    Computes the element-wise product of the two inputs over the first two
    direction
    Parameters:
    -----------
    S: np.ndarray
        The sensitivity maps of size [N,M,P,L]
    X: np.ndarray
        An image of size [N,M,P]
    Return:
    -------
    Sl = np.ndarray
        The product of every L element of S times X
    """
    Sl = np.copy(S)
    if Sl.shape == X.shape:
        for i in range(S.shape[3]):
            Sl[:, :, :, i] *= X[:, :, :, i]
    else:
        for i in range(S.shape[3]):
            Sl[:, :, :, i] *= X
    return Sl


def function_over_maps(f, x):
    """
    This methods computes the callable function over the third direction
    Parameters:
    -----------
    f: callable
        This function will be applyed n times where n is the last element in
        the shape of x
    x: np.ndarray
        Input data
    Return:
    -------
    np.list
        the results of the function as a list where the length of the list is
        equal to n
    """
    yl = []
    for i in range(x.T.shape[0]):
        yl.append(f((x.T[i]).T))
    return np.stack(yl, axis=len(yl[0].shape))
