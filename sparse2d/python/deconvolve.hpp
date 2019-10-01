#ifndef DECONVOLVE_H_
#define DECONVOLVE_H_

#include <iostream>

#include "sparse2d/IM_Obj.h"
#include "sparse2d/IM_IO.h"
#include "sparse2d/MR_Obj.h"
#include "sparse2d/IM_Deconv.h"
#include "sparse2d/MR_Deconv.h"
#include "sparse2d/MR_Sigma.h"
#include "sparse2d/MR_Sigma.h"

class MRDeconvolve
{
    public:
        MRDeconvolve(int type_of_deconvolution = 8,
                int type_of_multiresolution_transform = 2,
                int type_of_filters = 1,
                int number_of_undecimated_scales = -1,
                float sigma_noise = 0.,
                int type_of_noise = 1,
                int number_of_scales = DEFAULT_NBR_SCALE,
                float nsigma = DEFAULT_N_SIGMA,
                int number_of_iterations = DEFAULT_MAX_ITER_DECONV,
                float epsilon = DEFAULT_EPSILON_DECONV,
                bool psf_max_shift = true,
                bool verbose = false,
                bool optimization=false, 
                std::string residual_file_name = "",
                float fwhm_param = 0.,
                float convergence_param = 1.,
                float regul_param = 0.,
                std::string first_guess = "",
                std::string icf_filename = "",
                std::string rms_map = "",
                bool kill_last_scale=false,
                bool positive_constraint=false,
                bool keep_positiv_sup=false,
                bool sup_isol=false,
                float pas_codeur = -1, //FIXME: find default values 
                float sigma_gauss = -1,
                float mean_gauss = -1);
    //private:

};

MRDeconvolve::MRDeconvolve(int type_of_deconvolution, int type_of_multiresolution_transform,
                int type_of_filters, int number_of_undecimated_scales, float sigma_noise,
                int type_of_noise, int number_of_scales, float nsigma, int number_of_iterations,
                float epsilon, bool psf_max_shift, bool verbose, bool optimization, 
                std::string residual_file_name, float fwhm_param,
                float convergence_param, float regul_param, std::string first_guess,
                std::string icf_filename, std::string rms_map, bool kill_last_scale,
                bool positive_constraint, bool keep_positiv_sup, bool sup_isol,
                float pas_codeur, float sigma_gauss, float mean_gauss)
                {
                    std::cout << "it Works !" << std::endl;
                }
#endif