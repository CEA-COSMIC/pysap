# -*- coding: utf-8 -*-
##########################################################################
# PYSAP - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""


# Package import
import warnings
import numpy as np
from .utils import convert_locations_to_mask_3D
from pysap.plugins.mri.reconstruct.fourier import FourierBase
from modopt.interface.errors import warn

# Third party import
try:
    from pynufft import NUFFT_hsa
    from pynufft import NUFFT_cpu
except Exception:
    warnings.warn("pyNUFFT python package has not been found. Exists only "
                  "in Python3.x. Try Pynfft if non uniform Fourier"
                  " transform needed")


class InstanceCounter:
    """  InstanceCounter design pattern, to count the number of instanciation
    of a subclass.

    Attributes
    ----------
    numOfInstances: int
        number of instances of the InstanceCounter class.
    """

    numOfInstances = 0

    def countInstances(cls):
        """ This method increments the number of instantiations of the class
         associated with the cls obj"""
        cls.numOfInstances += 1

    # Insure the class is inherited in subclasses
    countInstances = classmethod(countInstances)

    def getNumInstances(cls):
        """ This method returns the number of instantiations of the class
         associated with the cls obj"""

        return cls.numOfInstances

    # Insure the class is inherited in subclasses
    getNumInstances = classmethod(getNumInstances)

    def __init__(self):
        """ Initialize a InstanceCounter object and immediatly increment the
        counter """

        self.countInstances()


class NUFFT(FourierBase, InstanceCounter):
    """  N-D non uniform Fast Fourrier Transform class

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (necessarly a square/cubic matrix).
    nufftObj: The pynufft object
        depending on the required computational platform
    platform: string, 'cpu', 'mcpu' or 'gpu'
        string indicating which hardware platform will be used to compute the
        NUFFT
    Kd: int or tuple
        int or tuple indicating the size of the frequency grid, for regridding.
        if int, will be evaluated to (Kd,)*nb_dim of the image
    Jd: int or tuple
        Size of the interpolator kernel. If int, will be evaluated
        to (Jd,)*dims image
    """
    numOfInstances = 0

    def __init__(self, samples, shape, platform='cpu', Kd=None, Jd=None):
        """ Initilize the 'NUFFT' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (necessarly a square/cubic matrix).
        platform: string, 'cpu', 'mcpu' or 'gpu'
            string indicating which hardware platform will be used to
            compute the NUFFT
        Kd: int or tuple
            int or tuple indicating the size of the frequency grid,
            for regridding. If int, will be evaluated
            to (Kd,)*nb_dim of the image
        Jd: int or tuple
            Size of the interpolator kernel. If int, will be evaluated
            to (Jd,)*dims image
        """
        self.shape = shape
        self.platform = platform
        self.samples = samples * (2 * np.pi)  # Pynufft use samples in
        # [-pi, pi[ instead of [-0.5, 0.5[
        self.dim = samples.shape[1]  # number of dimensions of the image

        if type(Kd) == int:
            self.Kd = (Kd,)*self.dim
        elif type(Kd) == tuple:
            self.Kd = Kd
        elif Kd is None:
            # Preferential option
            self.Kd = shape

        if type(Jd) == int:
            self.Jd = (Jd,)*self.dim
        elif type(Kd) == tuple:
            self.Jd = Jd
        elif Jd is None:
            # Preferential option
            self.Jd = (1,)*self.dim

        for (i, s) in enumerate(shape):
            assert(self.shape[i] <= self.Kd[i]), 'size of frequency grid' + \
                   'must be greater or equal than the image size'

        print('Creating the NUFFT object...')
        if self.platform == 'cpu':
            self.nufftObj = NUFFT_cpu()
            self.nufftObj.plan(self.samples, self.shape, self.Kd, self.Jd)

        elif self.platform == 'mcpu':
            warn('Attemping to use OpenCL plateform. Make sure to '
                 'have  all the dependecies installed')
            self.nufftObj = NUFFT_hsa()
            self.nufftObj.plan(self.samples, self.shape, self.Kd, self.Jd)
            self.nufftObj.offload('ocl')  # for multi-CPU computation

        elif self.platform == 'gpu':
            # In the case of GPU NUFFT, a bug in the CUDA implementation allows
            # us to only create one instance of the NUFFT_hsa object. The
            # CounterInstance metaclass of the NUFFT is here to insure the user
            # doesn't declare it more than once (in the GPU case)
            InstanceCounter.__init__(self)

            warn('Attemping to use Cuda plateform. Make sure to '
                 'have  all the dependecies installed and '
                 'to create only one instance of NUFFT GPU')
            if self.getNumInstances() > 1:
                raise RuntimeError('You have created more than one GPU NUFFT'
                                   ' object')
            self.nufftObj = NUFFT_hsa()
            self.nufftObj.plan(self.samples, self.shape, self.Kd, self.Jd)
            self.nufftObj.offload('cuda')  # for GPU computation

        else:
            raise ValueError('Wrong type of platform. Platform must be'
                             '\'cpu\', \'mcpu\' or \'gpu\'')

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as shape.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        if (self.platform == 'cpu'):
            y = self.nufftObj.forward(img)
        else:
            dtype = np.complex64
            # Send data to the mCPU/GPU platform
            self.nufftObj.x_Nd = self.nufftObj.thr.to_device(img.astype(dtype))
            gx = self.nufftObj.thr.copy_array(self.nufftObj.x_Nd)
            # Forward operator of the NUFFT
            gy = self.nufftObj.forward(gx)
            y = gy.get()

        return y * 1.0 / np.sqrt(np.prod(self.shape))

    def adj_op(self, x):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        if self.platform == 'cpu':
            img = self.nufftObj.adjoint(x)
        else:
            dtype = np.complex64
            cuda_array = self.nufftObj.thr.to_device(x.astype(dtype))
            gx = self.nufftObj.adjoint(cuda_array)
            img = gx.get()

        return img * np.sqrt(np.prod(self.shape))
