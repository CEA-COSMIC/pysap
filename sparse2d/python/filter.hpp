#ifndef FILTER_H_
#define FILTER_H_

#include <sparse2d/IM_Obj.h>
#include <sparse2d/IM_IO.h>
#include <sparse2d/MR_Obj.h>
#include <sparse2d/MR_Filter.h>
#include <sparse2d/MR_NoiseModel.h>
#include <sparse2d/MR_Abaque.h>
#include <sparse2d/MR_Psupport.h>
#include <sparse2d/NR.h>
#include <sparse2d/MR_Threshold.h>

#include <iomanip>

class MRFilters
{
    public:
        MRFilters(
            int type_of_filtering=1,
            double coef_detection_method=1,
            int type_of_multiresolution_transform=2,
            int type_of_filters = 1,
            int type_of_non_orthog_filters=2,
            int type_of_noise=1,
            int number_of_scales = DEFAULT_NBR_SCALE,
            float number_of_sigma = DEFAULT_N_SIGMA,
            double epsilon = DEFAULT_EPSILON_FILTERING,
            double iter_max = DEFAULT_MAX_ITER_FILTER,
            double max_inpainting_iter = DEFAULT_MAX_ITER_INPAINTING,
            std::string support_file_name="",
            float sigma_noise=0.,
            std::string flat_image="",
            std::string rms_map="",
            bool missing_data=false,
            bool keep_positiv_sup=false,
            bool write_info_on_prob_map=false,
            float epsilon_poisson = 1.00e-03,
            int size_block=7,
            int niter_sigma_clip=1,
            int first_scale=1,
            std::string mask_file_name="",
            std::string prob_mr_file="",
            int min_event_number=0,
            bool background_model_image=false,
            bool positive_recons_filter=false,
            bool suppress_isolated_pixels=false,
            bool verbose=false,
            int number_undec=-1,
            float pas_codeur=NULL,
            float sigma_gauss=NULL,
            float mean_gauss=NULL,
            bool old_poisson=false,
            bool positiv_ima= DEF_POSITIV_CONSTRAINT,
            bool max_ima = DEF_MAX_CONSTRAINT,
            bool kill_last_scale = false
            );
        
        void Info();
        void Filter();
        
    private:
        int type_of_filters;
        int number_of_scales;
        float number_of_sigma;
        double epsilon; //convergence parameter
        double iter_max;
        double max_inpainting_iter;
        std::string support_file_name;
        float sigma_noise;
        std::string flat_image;
        std::string rms_map;
        bool missing_data;
        bool keep_positiv_sup;
        bool write_info_on_prob_map;
        float epsilon_poisson;
        int size_block;
        int niter_sigma_clip;
        int first_scale;
        std::string mask_file_name;
        std::string prob_mr_file;
        int min_event_number;
        bool background_model_image;
        int number_undec;
        float pas_codeur;
        float sigma_gauss;
        float mean_gauss;
        bool positiv_ima;
        bool max_ima;
        bool kill_last_scale;
        bool verbose;
        

        type_noise stat_noise = DEFAULT_STAT_NOISE;
        type_transform transform = DEFAULT_MR_TRANS;
        type_filter filter = FILTER_THRESHOLD;
        type_threshold threshold = DEF_THRESHOLD;
        type_undec_filter undec_filter = DEF_UNDER_FILTER;
        type_sb_filter sb_filter = F_MALLAT_7_9;
};

MRFilters::MRFilters(
            int type_of_filtering,
            double coef_detection_method,
            int type_of_multiresolution_transform,
            int type_of_filters,
            int type_of_non_orthog_filters,
            int type_of_noise,
            int number_of_scales,
            float number_of_sigma,
            double epsilon,
            double iter_max,
            double max_inpainting_iter,
            std::string support_file_name,
            float sigma_noise,
            std::string flat_image,
            std::string rms_map,
            bool missing_data,
            bool keep_positiv_sup,
            bool write_info_on_prob_map,
            float epsilon_poisson,
            int size_block,
            int niter_sigma_clip,
            int first_scale,
            std::string mask_file_name,
            std::string prob_mr_file,
            int min_event_number,
            bool background_model_image,
            bool positive_recons_filter,
            bool suppress_isolated_pixels,
            bool verbose,
            int number_undec,
            float pas_codeur,
            float sigma_gauss,
            float mean_gauss,
            bool old_poisson,
            bool positiv_ima,
            bool max_ima,
            bool kill_last_scale
            )
{
    
    this->type_of_filters = type_of_filters;
    this->number_of_scales = number_of_scales;
    this->number_of_sigma = number_of_sigma;
    this->epsilon = epsilon;
    this->iter_max = iter_max;
    this->support_file_name = support_file_name;
    this->sigma_noise = sigma_noise;
    this->flat_image = flat_image;
    this->rms_map = rms_map;
    this->missing_data = missing_data;
    this->keep_positiv_sup = keep_positiv_sup;
    this->write_info_on_prob_map = write_info_on_prob_map;
    this->epsilon_poisson = (double)epsilon_poisson;
    this->size_block = size_block;
    this->niter_sigma_clip = niter_sigma_clip;
    this->first_scale = first_scale - 1;
    this->mask_file_name = mask_file_name;
    this->prob_mr_file = prob_mr_file;
    this->min_event_number = min_event_number;
    this->background_model_image = background_model_image;
    this->number_undec  = number_undec;
    this->pas_codeur = pas_codeur;
    this->sigma_gauss= sigma_gauss;
    this->mean_gauss = mean_gauss;
    this->max_inpainting_iter = max_inpainting_iter;
    this->positiv_ima = positiv_ima;
    this->max_ima = max_ima;
    this->kill_last_scale = kill_last_scale;
    this->verbose = verbose;


    if ((coef_detection_method > 0) && (coef_detection_method <= NBR_THRESHOLD))
        this->threshold = (type_threshold)(coef_detection_method-1);
    else
        throw std::invalid_argument("Error: bad type of detection.");

    //Verifier si il faut pas plutot mettre DEFAULT_N_SIGMA
    if (this->number_of_sigma < 0.) this->number_of_sigma = 0.1;

    if ((type_of_filtering > 0) && (type_of_filtering <= NBR_FILTERING))
        this->filter = (type_filter) (type_of_filtering-1);
    else
        throw std::invalid_argument("Error: bad type of filtering.");

    if ((type_of_multiresolution_transform > 0) && (type_of_multiresolution_transform <= NBR_TRANSFORM))
        this->transform = (type_transform) (type_of_multiresolution_transform-1);
    else
        throw std::invalid_argument("Error: bad type of transform.");

    if (this->type_of_filters != 1)
        this->sb_filter = get_filter_bank((char *)this->type_of_filters);

    if ((type_of_non_orthog_filters> 0) && (type_of_non_orthog_filters <= NBR_UNDEC_FILTER))
         this->undec_filter = (type_undec_filter) (type_of_non_orthog_filters-1);
    else
        throw std::invalid_argument("Error: bad type of filters.");

    if ((type_of_noise > 0) && (type_of_noise <= NBR_NOISE+1))
        this->stat_noise = (type_noise) (type_of_noise-1);
    else
        throw std::invalid_argument("Error: bad type of noise.");

    //check sigma noise
    if (this->sigma_noise != 0.)
        this->stat_noise = NOISE_GAUSSIAN;


    // Check the number of scales
    if ((this->number_of_scales <= 1) || (this->number_of_scales > MAX_SCALE))
        throw std::invalid_argument("Error: bad number of scale ]1;MAX_SCALE] ");

    //Nsigma a gerer le tableau chelou la

    if (this->iter_max <= 0)   this->iter_max = DEFAULT_MAX_ITER_FILTER;

    //check convergence parameter
    if ((this->epsilon < 0.) || (this->epsilon > 1.))
        this->epsilon = DEFAULT_EPSILON_FILTERING;
    
    if ((this->epsilon_poisson <= 0.) || (this->epsilon_poisson > MAX_EPSILON))
        throw std::invalid_argument("Error: bad precision number. [MIN_EPSILON;MAX_EPSILON]");
    
    if (this->size_block < 2)
        throw std::invalid_argument("Error: bad  SizeBlock parameter. SizeBlock > 1");

    if (this->niter_sigma_clip < 1)
        throw std::invalid_argument("Error: bad NiterClip parameter. NiterClip > 0 ");

    if (this->first_scale < 0)
        throw std::invalid_argument("Error: bad FirstScale parameter. FirstScale > 0");
    if (this->mask_file_name != "")
        this->missing_data = true;
    if (pas_codeur != NULL && mean_gauss != NULL && sigma_gauss != NULL)
    {
        this->stat_noise = NOISE_POISSON;
        this->sigma_noise = 1.;
    }

    if (this->number_of_sigma != DEFAULT_N_SIGMA && this->threshold == T_FDR)
        this->number_of_sigma = 2;
    
    if (this->stat_noise == NOISE_EVENT_POISSON)
    {
        if (this->transform != TO_PAVE_BSPLINE)
 	   {
 	      cerr << "WARNING: with this noise model, only the BSPLINE A TROUS can be used ... " << endl;

 	      cerr << "        Type transform is set to: BSPLINE A TROUS ALGORITHM " << endl;
 	   }
 	   this->transform = TO_PAVE_BSPLINE;
        if (this->number_of_scales == DEFAULT_NBR_SCALE)
 	        this->number_of_scales = DEF_N_SCALE;
 	   if (this->epsilon == DEFAULT_EPSILON_FILTERING)
            this->epsilon= DEF_EPS_EVENT_FILTERING;
    }

    if (this->stat_noise == NOISE_CORREL && this->rms_map == "")
        throw std::invalid_argument("Error: this noise model need a noise map (rms_map option)");

 	if (this->rms_map != "" && this->stat_noise != NOISE_NON_UNI_ADD && this->stat_noise !=  NOISE_CORREL)
 	   throw std::invalid_argument("Error: this noise model is not correct when RMS map option is set.");


	if (((SetTransform(this->transform) == TRANSF_MALLAT) ||
	       (SetTransform(this->transform) == TRANSF_FEAUVEAU))
 	      && ((this->stat_noise == NOISE_NON_UNI_ADD) ||
 	         (this->stat_noise  == NOISE_NON_UNI_MULT)))
  	    throw std::invalid_argument("  Error: with this transform, non stationary noise models are not valid : ");


       // isolated pixel removal is not valid with
       // non isotropic transform.
    if ((isotrop(this->transform) != True)
            && (suppress_isolated_pixels == True)
            && (this->transform != TO_UNDECIMATED_MALLAT)
            && (this->transform != TO_UNDECIMATED_NON_ORTHO))
        throw std::invalid_argument("Error: option suppress_isolated_pixels is not valid with non isotropic transform.");


       // Epsilon option and MaxIter option are valid only with iterative methods
      
    if ((this->epsilon != DEFAULT_EPSILON_FILTERING) || (this->iter_max != DEFAULT_MAX_ITER_FILTER))
    {   
       if ((this->filter != FILTER_ITER_THRESHOLD)
                && (this->filter != FILTER_ITER_ADJOINT_COEFF)
	            && (this->filter != FILTER_WAVELET_CONSTRAINT)
	            && (this->filter != FILTER_TV_CONSTRAINT))
            throw std::invalid_argument("Error: option -e and -i are not valid with non iterative filtering methods.");
    }

    if ((this->epsilon_poisson == 1.00e-03) //option is not set (default value)
            && (this->number_of_sigma != DEFAULT_N_SIGMA)) //option has been set
    {
        if (old_poisson)
	        this->epsilon_poisson = (1. - erf((double) this->number_of_sigma / sqrt((double) 2.))) / 2;
	    else
	        this->epsilon_poisson = erffc( (double) this->number_of_sigma / sqrt((double) 2.) ) / 2.;

	    if ((this->stat_noise == NOISE_EVENT_POISSON) && (this->number_of_sigma > 12))
	        throw std::invalid_argument("Error: number_of_sigma must be set to a lower value (<12).");
    }

 	if ((this->transform != TO_UNDECIMATED_MALLAT) && (this->transform != TO_MALLAT) && (type_of_filters != 1)) //Type_of_filters has been set
        throw std::invalid_argument("Error: option type_of_filters is only valid with Mallat transform");
}

void MRFilters::Filter()
{
    if (this->verbose == True)
    {
        cout << endl << endl << "PARAMETERS: " << endl << endl;

        //cout << "File Name in = " << Name_Imag_In << endl;
        //cout << "File Name Out = " << Name_Imag_Out << endl;
        cout << "Transform = " << StringTransform(this->transform) << endl;
        cout << "Number of scales = " << this->number_of_scales << endl;
        cout << "Noise Type = " << StringNoise(this->stat_noise) << endl;
        if ((this->transform == TO_MALLAT) || (this->transform == TO_UNDECIMATED_MALLAT))
        {
            cout << StringSBFilter(this->sb_filter) << endl;
            //if (Norm == NORM_L2) cout << "L2 normalization" << endl;
            if (this->transform == TO_UNDECIMATED_MALLAT)
                cout << "Number of undecimated scales = " <<  this->number_undec << endl;
        }
        if (this->transform == TO_UNDECIMATED_NON_ORTHO)
            cout << "Undec. Filter Bank = " << StringUndecFilter(this->undec_filter) << endl;
        if (this->stat_noise == NOISE_GAUSS_POISSON)
        {
           cout << "Type of Noise = POISSON" << endl;
           cout << "  Gain = " << this->pas_codeur << endl;
           cout << "  Read-out Noise Sigma  = " << this->sigma_gauss << endl;
           cout << "  Read-out Mean = " << this->mean_gauss << endl;
        }
        if ((this->stat_noise == NOISE_GAUSSIAN) && (this->sigma_noise > FLOAT_EPSILON))
            cout << "Sigma Noise = " << this->sigma_noise << endl;
        if (this->stat_noise ==  NOISE_EVENT_POISSON)
            cout << "Epsilon Poisson = " <<  this->epsilon_poisson << endl;
        if  (this->stat_noise == NOISE_SPECKLE)
            cout << "Epsilon speckle = " <<  this->epsilon_poisson << endl;
        if ((this->stat_noise !=  NOISE_EVENT_POISSON)
                && (this->stat_noise != NOISE_SPECKLE))
            cout << "N_Sigma = " << this->number_of_sigma << endl;

        cout << "Filter = " << StringFilter(this->filter) << endl;
        cout << "Epsilon = " << this->epsilon << endl;
        cout << "Max_Iter = " << this->iter_max << endl;
        if (this->max_inpainting_iter != DEFAULT_MAX_ITER_INPAINTING)
            cout << "Max_Inpainting_Iter = " << this->max_inpainting_iter << endl;
        if (this->positiv_ima == True) cout << "Positivity constraint" << endl;
        else cout << "No positivity constraint" << endl;

        if (this->max_ima == True) cout << "Maximum level constraint" << endl;
        else cout << "No maximum level constraint" << endl;

        if (this->keep_positiv_sup == True)
            cout << "Only positive wavelet coefficients are detected" << endl;
        if (this->first_scale > 0)
            cout << "Start the detect at scale " << this->first_scale + 1 << endl;
        if (this->kill_last_scale == True)
            cout << "Suppress the last scale" << endl;

        if (this->support_file_name != "")
            cout << "Support Image file name : " << this->support_file_name << endl;
}
}
void MRFilters::Info(){
    cout << "It works !";
 }

#endif