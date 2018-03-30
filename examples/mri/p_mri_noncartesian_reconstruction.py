"""
Neuroimaging non-cartesian Parallel reception reconstruction
============================================================

Credit: L Elgueddari, S.Lannuzel

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct.reconstruct import NFFT2
from pysap.plugins.mri.parallel_mri.utils import prod_over_maps
from pysap.plugins.mri.parallel_mri.utils import function_over_maps
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri.gradient import Grad_pMRI
# from pysap.plugins.mri.parallel_mri.gradient import Grad2D_pMRI_synthesis
from pysap.plugins.mri.parallel_mri.extract_sensitivity_maps import get_Smaps

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Loading input data
image_name = '/volatile/data/2017-05-30_32ch/'\
            '/meas_MID41_CSGRE_ref_OS1_FID14687.mat'
k_space_ref = loadmat(image_name)['ref']
k_space_ref /= np.linalg.norm(k_space_ref)
Smaps, SOS = get_Smaps(k_space_ref, (512, 512), mode='FFT')
mask = get_sample_data("mri-mask")
# mask.show()
image = pysap.Image(data=np.abs(SOS), metadata=mask.metadata)
image.show()


#############################################################################
# Generate the kspace
# -------------------

# Get the locations of the kspace samples and the associated observations
kspace_loc = convert_mask_to_locations(mask.data)

fourier_op_c = NFFT2(samples=kspace_loc, shape=image.shape)

# Generate the subsampled kspace
Sl = prod_over_maps(Smaps, SOS)
kspace_data = function_over_maps(fourier_op_c.op, Sl)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 20

linear_op = Wavelet2(wavelet_name="UndecimatedBiOrthogonalTransform",
                     nb_scale=4)

fourier_op = NFFT2(samples=kspace_loc, shape=(512, 512))
gradient_op = Grad_pMRI(data=kspace_data,
                        fourier_op=fourier_op,
                        linear_op=linear_op,
                        S=Smaps)

x_final, transform, cost = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    mu=1e-9,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=True)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()

plt.figure()
plt.plot(cost)
plt.show()
#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the CONDAT-VU reconstruction
max_iter = 10
gradient_op_cd = Grad_pMRI(data=kspace_data,
                           fourier_op=fourier_op,
                           S=Smaps)

x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    std_est=None,
    std_est_method=None,
    std_thr=2.,
    mu=0,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
