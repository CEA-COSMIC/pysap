"""
# Auto Thresholded cartesian reconstruction
# =========================================
# 
# Author: Chaithya G R / Pierre-Antoine Comby
# 
# In this tutorial we will reconstruct an MRI image from the sparse kspace
# measurements.
# 
# Import neuroimaging data
# ------------------------
# 
# We use the toy datasets available in pysap, more specifically a 2D brain slice
# and the cartesian acquisition scheme.
""" 
# %%
# Package import
import matplotlib.pyplot as plt
import numpy as np
from modopt.math.metrics import snr, ssim
from modopt.opt.linear import Identity
# Third party import
from modopt.opt.proximity import SparseThreshold
from mri.operators import FFT, WaveletN
from mri.operators.proximity.weighted import AutoWeightedSparseThreshold
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
from pysap.data import get_sample_data

image = get_sample_data('2d-mri')
print(image.data.min(), image.data.max())
image = image.data
image /= np.max(image)
mask = get_sample_data("cartesian-mri-mask")


# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(mask.data)
# Generate the subsampled kspace
fourier_op = FFT(mask=mask, shape=image.shape)
kspace_data = fourier_op.op(image)

# Zero order solution
image_rec0 = np.abs(fourier_op.adj_op(kspace_data))

# Calculate SSIM
base_ssim = ssim(image_rec0, image)
print(base_ssim)

#%%
# POGM optimization
# ------------------
# We now want to refine the zero order solution using an accelerated Proximal Gradient
# Descent algorithm (FISTA or POGM).
# The cost function is set to Proximity Cost + Gradient Cost


# Setup the operators
linear_op = WaveletN(wavelet_name="sym8", nb_scales=3)

# Manual tweak of the regularisation parameter
regularizer_op = SparseThreshold(Identity(), 2e-3, thresh_type="soft")
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)
# Start Reconstruction
x_final, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_data,
    optimization_alg='pogm',
    num_iterations=100,
    cost_op_kwargs={"cost_interval":None},
    metric_call_period=1,
    metrics = {
        "snr":{
            "metric": snr,
            "mapping": {"x_new":"test"},
            "cst_kwargs": {"ref": image},
            "early_stopping":False,
        },
        "ssim":{
            "metric": ssim,
            "mapping": {"x_new":"test"},
            "cst_kwargs": {"ref": image},
            "early_stopping": False,
        }
    }
)

image_rec = np.abs(x_final)
# image_rec.show()
# Calculate SSIM
recon_ssim = ssim(image_rec, image)
recon_snr= snr(image_rec, image)

print('The Reconstruction SSIM is : ' + str(recon_ssim))
print('The Reconstruction SNR is : ' + str(recon_snr))

#%%
# Threshold estimation using SURE 
# -------------------------------

_w = None

def static_weight(w, idx):
    print(np.unique(w))
    return w

# Setup the operators
linear_op = WaveletN(wavelet_name="sym8", nb_scale=3,padding_mode="periodization")
coeffs = linear_op.op(image_rec0)
print(linear_op.coeffs_shape)

# Here we don't manually setup the regularisation weights, but use statistics on the wavelet details coefficients

regularizer_op = AutoWeightedSparseThreshold(
    linear_op.coeffs_shape, linear=Identity(),
    update_period=0, # the weight is updated only once.
    sigma_range="global",
    thresh_range="global",
    threshold_estimation="sure",
    thresh_type="soft",
)
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)
# Start Reconstruction
x_final, costs, metrics2 = reconstructor.reconstruct(
    kspace_data=kspace_data,
    optimization_alg='pogm',
    num_iterations=100,
    metric_call_period=1,
    cost_op_kwargs={"cost_interval":None},
    metrics = {
         "snr":{
            "metric": snr,
            "mapping": {"x_new":"test"},
            "cst_kwargs": {"ref": image},
            "early_stopping":False,
        },
        "ssim":{
            "metric": ssim,
            "mapping": {"x_new":"test"},
            "cst_kwargs": {"ref": image},
            "early_stopping": False,
        },
        "cost_grad":{
            "metric": lambda x: reconstructor.gradient_op.cost(linear_op.op(x)),
            "mapping": {"x_new":"x"},
            "cst_kwargs": {},
            "early_stopping": False,
        },
        "cost_prox":{
            "metric": lambda x: reconstructor.prox_op.cost(linear_op.op(x)),
            "mapping": {"x_new":"x"},
            "cst_kwargs": {},
            "early_stopping": False,
        }
    }
)
image_rec2 = np.abs(x_final)
# image_rec.show()
# Calculate SSIM
recon_ssim2 = ssim(image_rec2, image)
recon_snr2 = snr(image_rec2, image)

print('The Reconstruction SSIM is : ' + str(recon_ssim2))
print('The Reconstruction SNR is : ' + str(recon_snr2))

plt.subplot(121)
plt.plot(metrics["snr"]["time"], metrics["snr"]["values"], label="pogm classic")
plt.plot(metrics2["snr"]["time"], metrics2["snr"]["values"], label="pogm sure global")
plt.ylabel("snr")
plt.xlabel("time")
plt.legend()
plt.subplot(122)
plt.plot(metrics["ssim"]["time"], metrics["ssim"]["values"])
plt.plot(metrics2["ssim"]["time"], metrics2["ssim"]["values"])
plt.ylabel("ssim")
plt.xlabel("time")
plt.figure()
plt.subplot(121)
plt.plot(metrics["snr"]["index"], metrics["snr"]["values"])
plt.plot(metrics2["snr"]["index"], metrics2["snr"]["values"])
plt.ylabel("snr")
plt.subplot(122)
plt.plot(metrics["ssim"]["index"], metrics["ssim"]["values"])
plt.plot(metrics2["ssim"]["index"], metrics2["ssim"]["values"])
plt.show()

#%%
# Qualitative results
# -------------------
#
def my_imshow(ax, img, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
            


fig, axs = plt.subplots(2,2)

my_imshow(axs[0,0], image, "Ground Truth")
my_imshow(axs[0,1], abs(image_rec0), f"Zero Order \n SSIM={base_ssim:.4f}")
my_imshow(axs[1,0], abs(image_rec), f"Fista Classic \n SSIM={recon_ssim:.4f}")
my_imshow(axs[1,1], abs(image_rec2), f"Fista Sure \n SSIM={recon_ssim2:.4f}")

fig.tight_layout()
plt.show()
