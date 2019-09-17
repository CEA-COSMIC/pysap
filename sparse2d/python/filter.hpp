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

#include "numpydata.hpp"

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
            float regul_param = DEFAULT_N_SIGMA,
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
            std::string background_model_image="",
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
            bool kill_last_scale = false,
            bool write_threshold = false
            );
        
        void Info();
        void noise_model_init(Ifloat data);
        void filtering_init(MRFiltering CFilter, Ifloat data);
        void Filter(py::array_t<float>& arr);
        
    private:
        int type_of_filters;
        int number_of_scales;
        float regul_param;
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
        std::string background_model_image;
        int number_undec;
        float pas_codeur;
        float sigma_gauss;
        float mean_gauss;
        bool positiv_ima;
        bool max_ima;
        bool kill_last_scale;
        bool verbose;
        bool write_threshold;
        bool old_poisson;
        bool suppress_isolated_pixels;
        bool positive_recons_filter;

        type_noise stat_noise = DEFAULT_STAT_NOISE;
        type_transform transform = DEFAULT_MR_TRANS;
        type_filter filter = FILTER_THRESHOLD;
        type_threshold threshold = DEF_THRESHOLD;
        type_undec_filter undec_filter = DEF_UNDER_FILTER;
        type_sb_filter sb_filter = F_MALLAT_7_9;
        MultiResol mr;
        MRNoiseModel model_data;
        FilterAnaSynt fas;
        FilterAnaSynt *ptrfas = NULL;
};

MRFilters::MRFilters(
            int type_of_filtering,
            double coef_detection_method,
            int type_of_multiresolution_transform,
            int type_of_filters,
            int type_of_non_orthog_filters,
            int type_of_noise,
            int number_of_scales,
            float regul_param,
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
            std::string background_model_image,
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
            bool kill_last_scale,
            bool write_threshold
            )
{
    
    this->type_of_filters = type_of_filters;
    this->number_of_scales = number_of_scales;
    this->regul_param = regul_param;
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
    this->write_threshold = write_threshold;
    this->suppress_isolated_pixels = suppress_isolated_pixels;
    this->positive_recons_filter = positive_recons_filter;

    if ((coef_detection_method > 0) && (coef_detection_method <= NBR_THRESHOLD))
        this->threshold = (type_threshold)(coef_detection_method-1);
    else
        throw std::invalid_argument("Error: bad type of detection.");

    //Verifier si il faut pas plutot mettre DEFAULT_N_SIGMA
    if (this->regul_param < 0.) this->regul_param = 0.1;

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

    if (this->regul_param != DEFAULT_N_SIGMA && this->threshold == T_FDR)
        this->regul_param = 2;
    
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
            && (this->regul_param != DEFAULT_N_SIGMA)) //option has been set
    {
        if (old_poisson)
	        this->epsilon_poisson = (1. - erf((double) this->regul_param / sqrt((double) 2.))) / 2;
	    else
	        this->epsilon_poisson = erffc( (double) this->regul_param / sqrt((double) 2.) ) / 2.;

	    if ((this->stat_noise == NOISE_EVENT_POISSON) && (this->regul_param > 12))
	        throw std::invalid_argument("Error: regul_param must be set to a lower value (<12).");
    }

 	if ((this->transform != TO_UNDECIMATED_MALLAT) && (this->transform != TO_MALLAT) && (type_of_filters != 1)) //Type_of_filters has been set
        throw std::invalid_argument("Error: option type_of_filters is only valid with Mallat transform");
}

static inline double dierfc(double y)
{
    double s, t, u, w, x, z;

    z = y;
    if (y > 1) {
        z = 2 - y;
    }
    w = 0.916461398268964 - log(z);
    u = sqrt(w);
    s = (log(u) + 0.488826640273108) / w;
    t = 1 / (u + 0.231729200323405);
    x = u * (1 - s * (s * 0.124610454613712 + 0.5)) -
        ((((-0.0728846765585675 * t + 0.269999308670029) * t +
        0.150689047360223) * t + 0.116065025341614) * t +
        0.499999303439796) * t;
    t = 3.97886080735226 / (x + 3.97886080735226);
    u = t - 0.5;
    s = (((((((((0.00112648096188977922 * u +
        1.05739299623423047e-4) * u - 0.00351287146129100025) * u -
        7.71708358954120939e-4) * u + 0.00685649426074558612) * u +
        0.00339721910367775861) * u - 0.011274916933250487) * u -
        0.0118598117047771104) * u + 0.0142961988697898018) * u +
        0.0346494207789099922) * u + 0.00220995927012179067;
    s = ((((((((((((s * u - 0.0743424357241784861) * u -
        0.105872177941595488) * u + 0.0147297938331485121) * u +
        0.316847638520135944) * u + 0.713657635868730364) * u +
        1.05375024970847138) * u + 1.21448730779995237) * u +
        1.16374581931560831) * u + 0.956464974744799006) * u +
        0.686265948274097816) * u + 0.434397492331430115) * u +
        0.244044510593190935) * t -
        z * exp(x * x - 0.120782237635245222);
    x += s * (x * s + 1);
    if (y > 1) {
        x = -x;
    }
    return x;
}
static inline char* to_char(std::string str) {
    return const_cast<char*>(str.c_str());
}

void MRFilters::noise_model_init(Ifloat data)
{
    if ((this->transform == TO_MALLAT) || (this->transform == TO_UNDECIMATED_MALLAT)) {
        fas.Verbose = (Bool)this->verbose;
        fas.alloc(this->sb_filter);
        ptrfas = &fas;
    }
    model_data.set_old_poisson((Bool)this->old_poisson);
    model_data.write_threshold((Bool)this->write_threshold);
    model_data.alloc(this->stat_noise, data.nl(), data.nc(), 
                    this->number_of_scales, this->transform, ptrfas, NORM_L1, this->number_undec);

    int number_band = model_data.nbr_band();

    if (this->sigma_noise > FLOAT_EPSILON)
        model_data.SigmaNoise = this->sigma_noise;
    /*
    if (UseNSigma  == True)
    {
        double n_sigma_12 = erffc( (double) 12. / sqrt((double) 2.) ) / 2.;
    	if(Verbose) cout << " NbrTabNSigma = " << NbrTabNSigma << endl;
    	 if (NbrTabNSigma == 0) for (i=0; i < number_band; i++) model_data.NSigma[i]=N_Sigma;
        else
        {
        	  for (i=0; i < NbrTabNSigma ; i++)   model_data.NSigma[i]=TabNSigma[i];
        	  for (i=NbrTabNSigma ; i < NbrBand; i++)   model_data.NSigma[i]=TabNSigma[NbrTabNSigma-1];
        }
        for (s=0; s <  number_band; s++) model_data.TabEps[s] = (number_band.NSigma[s] < 12) ? erffc( (double)  number_band.NSigma[s] / sqrt((double) 2.) ) / 2.: n_sigma_12;
    }
    else for (s=0; s <  number_band; s++) model_data.TabEps[s] = this->epsilon_poisson;

    if (N_Sigma  == 111)
    {
        for (i=0; i < 4; i++)  number_band.NSigma[i] = 5;
	    model_data.NSigma[4] = 4.;
	    model_data.NSigma[5] = 3.5;
 	    for (i=6; i < NbrBand; i++) model_data.NSigma[i] = 3.;
    }*/
    if (this->suppress_isolated_pixels)
        model_data.SupIsol = True;

    model_data.OnlyPositivDetect = (Bool)this->keep_positiv_sup;
    model_data.NiterSigmaClip = this->niter_sigma_clip;
    model_data.SizeBlockSigmaNoise = this->size_block;
    model_data.FirstDectectScale = this->first_scale;
    model_data.CCD_Gain = this->pas_codeur;
    model_data.CCD_ReadOutSigma = this->sigma_gauss;
    model_data.CCD_ReadOutMean = this->mean_gauss;
    model_data.TypeThreshold = this->threshold;
    model_data.U_Filter = this->undec_filter;

    if (this->min_event_number > 0)
        model_data.MinEventNumber = this->min_event_number;

    if (this->rms_map != ""){
       model_data.UseRmsMap = True;
       io_read_ima_float(to_char(this->rms_map), model_data.RmsMap);
    }
}

void MRFilters::filtering_init(MRFiltering CFilter, Ifloat data)
{
    int i,j;
    if (this->background_model_image != "") // i.e. if (use_background_model_image == True)
    {
       io_read_ima_float(to_char(this->background_model_image), CFilter.BackgroundData);
       if ((CFilter.BackgroundData.nl() != data.nl()) || (CFilter.BackgroundData.nc() != data.nc()))
           throw std::invalid_argument("Error: the background image must have the same size as the input data.");
       CFilter.BackgroundImage = True;
    }

    CFilter.MissingData = (Bool)this->missing_data;
    if (this->missing_data)
    {
        if (this->mask_file_name != "") // i.e. if (use_mask_file == True)
        {
            (CFilter.MaskIma).alloc(data.nl(), data.nc(), "mask");
            for (i=0; i < data.nl(); i++)
                for (j=0; j < data.nc(); j++)
                    (CFilter.MaskIma)(i,j) = (data(i,j) == 0) ? 0: 1;
        }
        else
        {
            io_read_ima_float(to_char(this->mask_file_name), CFilter.MaskIma);
            if ((CFilter.MaskIma.nl() != data.nl()) || (CFilter.MaskIma.nc() != data.nc() ))
                throw std::invalid_argument("Error: Mask size = " + std::to_string(CFilter.MaskIma.nl()) + ", "
                + std::to_string(CFilter.MaskIma.nc()) + " Data size = " + std::to_string(data.nl()) + ", " + std::to_string(data.nc()));
        }
    }
    //est-ce que les if sont vrmts necessaires?

    if (this->max_inpainting_iter != DEFAULT_MAX_ITER_INPAINTING) // i.e. MaxInpaintingIterOpt == True
        CFilter.Max_Inpainting_Iter = this->max_inpainting_iter;
    if (this->iter_max != DEFAULT_MAX_ITER_FILTER)
        CFilter.Max_Iter = this->iter_max;
    if (this->kill_last_scale)
        CFilter.KillLastScale = (Bool)this->kill_last_scale;
    if (this->epsilon != DEFAULT_EPSILON_FILTERING)
        CFilter.Epsilon = this->epsilon;
    if (this->positiv_ima != DEF_POSITIV_CONSTRAINT)
        CFilter.PositivIma = (Bool)this->positiv_ima;
    if (this->max_ima)
        CFilter.MaxIma = True;
  
    CFilter.Verbose = (Bool)this->verbose;
    if ((this->regul_param != 0.1) || (this->filter == FILTER_TV_CONSTRAINT)) //default value=0.1
        CFilter.RegulParam = this->regul_param;
    if (this->flat_image != "") {
        CFilter.UseFlatField = True;
        io_read_ima_float(to_char(this->flat_image), CFilter.FlatField);
    }
    else
        CFilter.UseFlatField = False;
    
    if (this->positive_recons_filter)
        CFilter.Sup_Set = False;
}

void MRFilters::Filter(py::array_t<float>& arr)
{
    int s,k;
    if (this->verbose == True)
        Info();
    
    //Soft.sparse2d(); what is this?
    #ifndef MRPOL
        lm_check(LIC_MR1);
    #else 
        lm_check(LIC_POL);
    #endif

    Ifloat data = array2image_2d(arr);
    Ifloat Result(data.nl(), data.nc(), (char *) "Result Filtering");
    check_scale(data.nl(), data.nc(), this->number_of_scales);

    // noise model class initialization
    noise_model_init(data);

    // Filtering class initialization
    MRFiltering CFilter(model_data, this->filter);
    filtering_init(CFilter, data);

    //perform the filter operation
    CFilter.filter(data, Result);

    if (this->stat_noise == NOISE_EVENT_POISSON)
        //Header.bitpix = BP_FLOAT;

    //io_write_ima_float(Name_Imag_Out, Result, &Header);

    // support image creation
    if ((this->support_file_name != "") && (this->number_of_scales > 1))
        model_data.write_support_mr(to_char(this->support_file_name));

    if ((this->verbose) && (this->stat_noise == NOISE_GAUSSIAN)  && (ABS(this->sigma_noise) < FLOAT_EPSILON))
         cout << "Noise standard deviation = " <<  model_data.SigmaNoise << endl;

    //write info used for computing the probability map
    if ((this->stat_noise == NOISE_EVENT_POISSON) && (this->prob_mr_file != ""))
    {
        int b,i,j;
	    if(this->write_info_on_prob_map)
            model_data.write_in_few_event(True) ;

        //MultiResol mr;
        //io_read_ima_float(Name_Imag_In, data, &Header);
        mr.alloc(data.nl(), data.nc(),this->number_of_scales,model_data.type_trans(),
                     model_data.filter_bank(), model_data.TypeNorm, model_data.nbr_undec_scale(), model_data.U_Filter);

       if (model_data.TransImag)
            model_data.im_transform(data);
        
       int last_scale = mr.nbr_band() - 1;

       if (this->background_model_image != "")
            data -= CFilter.BackgroundData;
        
        mr.transform(data);

       if (this->keep_positiv_sup)
       {
          for (b = 0; b < last_scale; b++)
          for (i = 0; i < mr.size_band_nl(b); i++)
          for (j = 0; j < mr.size_band_nc(b); j++)
             if (mr(b,i,j) < 0) mr(b,i,j) = 0.;
       }

        // Last scale contains the max SNR through the scales
        b = last_scale;
        for (i=0; i< mr.size_band_nl(b); i++)
        for (j=0; j< mr.size_band_nc(b); j++)
            mr(b,i,j)=0;

       for (b = 0; b < last_scale; b++)
       for (i = 0; i < mr.size_band_nl(b); i++)
       for (j = 0; j < mr.size_band_nc(b); j++)
       {
           double Coef = model_data.prob_signal_few_event( mr(b,i,j),  b, i, j ) ;
           if (b < last_scale)
           {
               if (Coef < 1e-35)
                    Coef = (float) (sqrt((double) 2.) * fabs(dierfc((1e-35))));
               else
                    Coef = sqrt(2.) * ABS(dierfc(Coef));
               mr(b,i,j) = Coef;
 	       }
           if (mr(b,i,j) > mr(last_scale,i,j))
                mr(last_scale,i,j) = mr(b,i,j);
       }
       mr.write(to_char(this->prob_mr_file));
    }
}

void MRFilters::Info(){
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
        cout << "N_Sigma = " << this->regul_param << endl;

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

#endif
