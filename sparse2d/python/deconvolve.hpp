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
#include "filter.hpp"

#define NBR_OK_METHOD 5


class MRDeconvolve
{
    public:
        MRDeconvolve(int type_of_deconvolution = 3,
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
                float fwhm_param = 0.,
                float convergence_param = 1.,
                float regul_param = 0.,
                std::string first_guess = "",
                std::string icf_filename = "",
                std::string rms_map = "",
                bool kill_last_scale=false,
                bool positive_constraint=true,
                bool keep_positiv_sup=false,
                bool sup_isol=false,
                float pas_codeur = 1, 
                float sigma_gauss = 0,
                float mean_gauss = 0);

                void Info();
                py::array_t<float> Deconvolve(py::array_t<float>& arr, py::array_t<float>& psf);
                void DeconvInit();
                void NoiseModelInit();

    private:
        float convergence_param;
        float regul_param;
        int number_of_undecimated_scales;
        std::string first_guess;
        std::string icf_filename;
        std::string rms_map;
        bool kill_last_scale;
        bool positive_constraint;
        bool keep_positiv_sup;
        bool sup_isol;
        float sigma_noise;
        float nsigma;
        int number_of_iterations;
        bool psf_max_shift;
        bool verbose;
        bool optimization;
        float fwhm_param;
        float pas_codeur;
        float sigma_gauss;
        float mean_gauss;
        int number_of_scales;
        float epsilon;

        MRDeconv CDec;
        MRNoiseModel model_data;
        bool gauss_conv = false;
        bool use_nsigma = false;
        type_sb_filter sb_filter = F_MALLAT_7_9;
        type_noise stat_noise = DEFAULT_STAT_NOISE;
        type_transform transform = DEFAULT_TRANSFORM;
        type_deconv deconv = DEFAULT_DECONV;
        type_deconv TabMethod[NBR_OK_METHOD] = {DEC_MR_CITTERT,DEC_MR_GRADIENT,
                            DEC_MR_LUCY, DEC_MR_MAP,DEC_MR_VAGUELET};

};

MRDeconvolve::MRDeconvolve(int type_of_deconvolution, int type_of_multiresolution_transform,
                int type_of_filters, int number_of_undecimated_scales, float sigma_noise,
                int type_of_noise, int number_of_scales, float nsigma, int number_of_iterations,
                float epsilon, bool psf_max_shift, bool verbose, bool optimization, 
                float fwhm_param, float convergence_param, float regul_param, std::string first_guess,
                std::string icf_filename, std::string rms_map, bool kill_last_scale,
                bool positive_constraint, bool keep_positiv_sup, bool sup_isol,
                float pas_codeur, float sigma_gauss, float mean_gauss)
                {
                    this->convergence_param = convergence_param;
                    this->regul_param = regul_param;
                    this->number_of_undecimated_scales = number_of_undecimated_scales;
                    this->first_guess = first_guess;
                    this->icf_filename = icf_filename;
                    this->rms_map =rms_map;
                    this->kill_last_scale = kill_last_scale;
                    this->positive_constraint = positive_constraint;
                    this->keep_positiv_sup = keep_positiv_sup;
                    this->sup_isol = sup_isol;
                    this->sigma_noise = sigma_noise;
                    this->nsigma = nsigma;
                    this->number_of_iterations = number_of_iterations;
                    this->psf_max_shift = psf_max_shift;
                    this->optimization = optimization;
                    this->verbose = verbose;
                    this->number_of_scales = number_of_scales;
                    this->fwhm_param = fwhm_param;
                    this->pas_codeur = pas_codeur;
                    this->sigma_gauss = sigma_gauss;
                    this->mean_gauss = mean_gauss;

                    if (this->fwhm_param > 0)
                        this->gauss_conv = True;
                    
                    if (this->gauss_conv && this->icf_filename != "")
                        throw std::invalid_argument("Error: icf_filename and fwhm_param options are not compatible ..");

                    if ((epsilon < 0) || (epsilon > 1.)) 
                        this->epsilon = DEFAULT_EPSILON_DECONV;
                    else
                        this->epsilon = epsilon;

                    if (type_of_deconvolution <= 0 || type_of_deconvolution > NBR_OK_METHOD)
                        throw std::invalid_argument("Error: bad type of deconvolution: " + std::to_string(type_of_deconvolution));
                    else if (type_of_deconvolution != 3)                
		                this->deconv  = (type_deconv) (TabMethod[type_of_deconvolution-1]);

                    if (type_of_noise > 0 && type_of_noise < NBR_NOISE && type_of_noise != 1)
                        this->stat_noise = (type_noise) (type_of_noise-1);
                    else if (type_of_noise != 1)
                        throw std::invalid_argument("Error: bad type of noise: " + std::to_string(type_of_noise));
                    
                    if (type_of_multiresolution_transform != 2 &&
                            type_of_multiresolution_transform > 0 &&
                            type_of_multiresolution_transform <= NBR_TRANSFORM)
                        this->transform = (type_transform)(type_of_multiresolution_transform-1);
                    else if (type_of_multiresolution_transform != 2)
                        throw std::invalid_argument("Error: bad type of transform: " + std::to_string(type_of_multiresolution_transform));
                    
                    if (type_of_filters != 1)
                        this->sb_filter = get_filter_bank(int_to_char(type_of_filters));
                    
                    if (sigma_noise != 0.)
                        this->stat_noise = NOISE_GAUSSIAN;
                    
                    if (this->sigma_gauss != 0 && this->pas_codeur != 1 && this->mean_gauss != 0)
                    {
                        this->stat_noise = NOISE_POISSON;
                        this->sigma_noise = 1.;
                    }
                    if ((number_of_scales <= 1 || number_of_scales > MAX_SCALE) && number_of_scales != DEFAULT_NBR_SCALE)
                        throw std::invalid_argument("Error: bad number of scales: ]1; " + std::to_string(MAX_SCALE) + "]");
                    
                    if (this->nsigma != DEFAULT_N_SIGMA)
                    {
                        this->use_nsigma = true;
                        if (nsigma <= 0.)
                            this->nsigma = DEFAULT_N_SIGMA;
                    }
                    if (number_of_iterations <= 0)
                        this->number_of_iterations = DEFAULT_MAX_ITER_DECONV;

                    if (this->regul_param  < 0.)
                        this->regul_param = 0.1;

                    if (type_of_deconvolution <= 0 || type_of_deconvolution > NBR_OK_METHOD)
                        throw std::invalid_argument("Error: bad type of deconvolution: " + std::to_string(type_of_deconvolution));
                

                    if ((this->rms_map != "") && (this->stat_noise != NOISE_NON_UNI_ADD) && (this->stat_noise !=  NOISE_CORREL))
                        throw std::invalid_argument(std::string("Error: this noise model is not correct when RMS map option is set.\n       Valid models are: \n        ") +
                                                    + StringNoise(NOISE_NON_UNI_ADD) + "        " + StringNoise(NOISE_CORREL));
                    
                    if ((this->stat_noise ==  NOISE_CORREL) && (this->rms_map == ""))
                        throw std::invalid_argument("Error: this noise model needs a noise map (rms_map option) ");
	
                    if ((isotrop(transform) == False)
                            && ((this->stat_noise == NOISE_NON_UNI_ADD) ||
                                (this->stat_noise  == NOISE_NON_UNI_MULT)))
                        throw std::invalid_argument(std::string("Error: with this transform, non stationary noise models are not valid : ") + StringFilter(FILTER_THRESHOLD));
                 
                    if (this->kill_last_scale && this->optimization)
                        throw std::invalid_argument("Error: kill_last_scale and optimization options are not compatible ... ");

                    if (this->kill_last_scale && ((this->deconv == DEC_MR_LUCY) || (this->deconv ==DEC_MR_MAP)))
                            throw std::invalid_argument("Error: kill_last_scale option cannot be used with this deconvolution method  ... ");
	   
	
                    if ((this->number_of_iterations == DEFAULT_MAX_ITER_DECONV) && (this->deconv == DEC_MR_VAGUELET))
                        this->number_of_iterations = 10;

                    if ((this->transform != TO_UNDECIMATED_MALLAT) && (this->transform != TO_MALLAT) && (type_of_filters != 1))
                        throw std::invalid_argument("Error: option type_of_filters is only valid with Mallat transform ...");

                    if ((convergence_param == 1.) && (this->regul_param > 1.))
                        this->convergence_param = 1. / (2. * this->regul_param);
                }

void MRDeconvolve::Info()
{
    cout << endl << endl << "PARAMETERS: " << endl << endl;
    cout << "Transform = " << StringTransform(this->transform) << endl;
    cout << "Number of scales = " << this->number_of_scales << endl;
    if (this->stat_noise == NOISE_GAUSSIAN) {
        cout << "Type of Noise = GAUSSIAN" << endl;
        if (this->sigma_noise > 0)
            cout << "Sigma Noise = " << this->sigma_noise << endl;
    }
    else {
        cout << "Type of Noise = POISSON" << endl;
        cout << "  Gain = " << this->pas_codeur << endl;
        cout << "  Read-out Noise Sigma  = " << this->sigma_gauss << endl;
        cout << "  Read-out Mean = " << this->mean_gauss << endl;
    }
    cout << "Deconv = " << StringDeconv(this->deconv) << endl;
    cout << "N_Sigma = " << this->nsigma << endl;
    cout << "Epsilon = " << this->epsilon << endl;
    cout << "Max_Iter = " << this->number_of_iterations << endl;
    cout << "Convergence paramter = " << this->convergence_param << endl;
    if (this->kill_last_scale)
        cout << "Kill last scale " << endl;
    cout << "Fwhm = " << this->fwhm_param << endl;
}

void MRDeconvolve::DeconvInit()
{
    this->CDec.KillLastScale = (Bool)this->kill_last_scale;
    this->CDec.PositivConstraint = (Bool)this->positive_constraint;
    this->CDec.DecMethod = this->deconv;
    this->CDec.PsfMaxShift = (Bool)this->psf_max_shift;
    this->CDec.Noise_Ima = this->sigma_noise;
    this->CDec.MaxIter = this->number_of_iterations;
    this->CDec.EpsCvg = this->epsilon;
    this->CDec.IterCvg = this->convergence_param;
    this->CDec.GaussConv = (Bool)this->gauss_conv;
    this->CDec.Fwhm = this->fwhm_param;
    this->CDec.OptimParam = (Bool)this->optimization;
    this->CDec.Verbose = (Bool)this->verbose;
    this->CDec.RegulParam = this->regul_param;
    this->CDec.StatNoise = this->stat_noise;
}

void MRDeconvolve::NoiseModelInit()
{
    int nbr_band = model_data.nbr_band();
    model_data.OnlyPositivDetect = (Bool)this->keep_positiv_sup;

    if (this->sigma_noise > FLOAT_EPSILON)
        model_data.SigmaNoise = this->sigma_noise;

    if (this->nsigma != DEFAULT_N_SIGMA)
    {
            for (int b = 0; b < nbr_band; b++)
            model_data.NSigma[b] = this->nsigma;
    }

    model_data.NiterSigmaClip = 1;
    model_data.SizeBlockSigmaNoise = DEFAULT_SIZE_BLOCK_SIG;
    model_data.CCD_Gain = this->pas_codeur;
    model_data.CCD_ReadOutSigma = this->sigma_gauss;
    model_data.CCD_ReadOutMean = this->mean_gauss;

    if (this->sup_isol)
        model_data.SupIsol = True;

    if (this->rms_map != "") {
        model_data.UseRmsMap = True;
        io_read_ima_float(to_char(this->rms_map), model_data.RmsMap);
    }
}

py::array_t<float> MRDeconvolve::Deconvolve(py::array_t<float>& arr, py::array_t<float>& psf)
{
    Ifloat Guess, Ima_ICF;
    Ifloat *Pt_G = NULL;
    Ifloat *Pt_ICF = NULL;

    //outputs information
    if (this->verbose)
        Info();

    //read input image 
    this->CDec.Imag = array2image_2d(arr);
    this->CDec.Psf = array2image_2d(psf);

    //read additional files
    if (this->first_guess != "") {
        io_read_ima_float(to_char(this->first_guess), Guess);
        if (this->first_guess != "") Pt_G = &Guess;
    }
    if (this->icf_filename != "") {
        io_read_ima_float(to_char(this->icf_filename), Ima_ICF);
        Pt_ICF = &Ima_ICF;
    }

    //deconvolution class initialization
    DeconvInit();

    if (this->verbose)
        cout << " Start the deconvolution ... " << endl;
    
    //noise model class initialization
    FilterAnaSynt FAS;
    FilterAnaSynt *PtrFAS = NULL;

    if ((this->transform == TO_MALLAT) || (this->transform == TO_UNDECIMATED_MALLAT)) {
        FAS.Verbose = (Bool)this->verbose;
        FAS.alloc(this->sb_filter);
        PtrFAS = &FAS;
    }
    model_data.alloc(this->stat_noise, CDec.Imag.nl(), CDec.Imag.nc(),this->number_of_scales,
                        this->transform, PtrFAS, NORM_L1, this->number_of_undecimated_scales);
    NoiseModelInit();
    this->CDec.ModelData = &model_data;

    //deconvolution
    this->CDec.im_deconv(Pt_G, Pt_ICF);

    return image2array_2d(this->CDec.Obj);
}


#endif