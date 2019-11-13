/*##########################################################################
pySAP - Copyright (C) CEA, 2017 - 2018
Distributed under the terms of the CeCILL-B license, as published by
the CEA-CNRS-INRIA. Refer to the LICENSE file or to
http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
for details.
##########################################################################*/

#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Includes
#include <iostream>
#include <string>
#include <sstream>
#include <sparse2d/IM_Obj.h>
#include <sparse2d/IM_IO.h>
#include <sparse2d/MR_Obj.h>
#include <sparse2d/IM_Prob.h>
#include "numpydata.hpp"


class MRTransform {

public:
    // Constructor
    MRTransform(
        int type_of_multiresolution_transform,
        int type_of_lifting_transform=3,
        int number_of_scales=4,
        int iter=3,
        int type_of_filters=1,
        bool use_l2_norm=false,
        int type_of_non_orthog_filters=2,
        int bord=0,
        int nb_procs=0,
        int verbose=0);

    // Destructor
    ~MRTransform();

    // Save transformation method
    void Save(MultiResol &mr);

    // Update transformation method
    MultiResol Update(int nl, int nc);

    // Information method
    void Info();

    // Transform method
    py::list Transform(py::array_t<float>& arr, bool save=false);

    // Reconstruction method
    py::array_t<float> Reconstruct(py::list mr_data);

    // Getter/setter functions for the input/output image path
    void set_opath(std::string path) {this->m_opath = path;}
    string get_opath() const {return m_opath;}

private:
    MultiResol mr;
    FilterAnaSynt fas;
    FilterAnaSynt *ptrfas = NULL;
    bool mr_initialized;
    std::string m_opath;
    int type_of_multiresolution_transform;
    int type_of_lifting_transform;
    int number_of_scales;
    int iter;
    int type_of_filters;
    bool use_l2_norm;
    int type_of_non_orthog_filters;
    int nb_procs;
    int verbose;
    type_transform mr_transform = DEFAULT_TRANSFORM;
    type_lift lift_transform = DEF_LIFT;
    type_sb_filter filter = F_MALLAT_7_9;
    sb_type_norm norm = NORM_L1;
    type_undec_filter no_filter = DEF_UNDER_FILTER;
    type_border bord;
    int nb_of_undecimated_scales;
};

// Constructor
MRTransform::MRTransform(
        int type_of_multiresolution_transform,
        int type_of_lifting_transform,
        int number_of_scales,
        int iter,
        int type_of_filters,
        bool use_l2_norm,
        int type_of_non_orthog_filters,
        int bord,
        int nb_procs,
        int verbose){
    // Define instance attributes
    switch (bord)
    {
      case 0: this->bord = I_ZERO; break;
      case 1: this->bord = I_CONT; break;
      case 2: this->bord = I_MIRROR; break;
      case 3: this->bord = I_PERIOD; break;
      default:
         throw std::invalid_argument("Error: bad parameter bord.");
    }
    this->bord = I_CONT;
    this->type_of_multiresolution_transform = type_of_multiresolution_transform;
    this->type_of_lifting_transform = type_of_lifting_transform;
    this->number_of_scales = number_of_scales;
    this->iter = iter;
    this->type_of_filters = type_of_filters;
    this->use_l2_norm = use_l2_norm;
    this->type_of_non_orthog_filters = type_of_non_orthog_filters;
    this->verbose = verbose;
    this->nb_of_undecimated_scales = -1;
    this->mr_initialized = false;

    // The maximum number of threads returned by omp_get_max_threads()
    // (which is the default number of threads used by OMP in parallel
    // regions) can sometimes be far below the number of CPUs.
    // It is then required to set it in relation to the real number of CPUs
    // (the -1 is used to live one thread to the main process which ensures
    // better and more constant performances). - Fabrice Poupon 2013/03/09
    #ifdef _OPENMP
        if (nb_procs <= 0)
            this->nb_procs = omp_get_num_procs() - 1;
        else
            this->nb_procs = nb_procs;
        omp_set_num_threads(this->nb_procs);
    #endif

    // Load the mr transform
    if ((this->type_of_multiresolution_transform > 0) && (this->type_of_multiresolution_transform <= NBR_TOT_TRANSFORM+1))
        this->mr_transform = type_transform(this->type_of_multiresolution_transform - 1);
    else
        throw std::invalid_argument("Invalid MR transform number.");

    // Load the lifting transform
    if ((this->type_of_lifting_transform > 0) && (this->type_of_lifting_transform <= NBR_LIFT))
        this->lift_transform = type_lift(this->type_of_lifting_transform);
    else
        throw std::invalid_argument("Invalid lifting transform number.");

    // Check the number of scales
    if ((this->number_of_scales <= 1) || (this->number_of_scales > MAX_SCALE))
        throw std::invalid_argument("Bad number of scales ]1; MAX_SCALE].");

    // Check the number of iterations
    if ((this->iter <= 1) || (this->iter > 20))
        throw std::invalid_argument("Bad number of iteration ]1; 20].");

    // Load the filter
    if (this->type_of_filters != 1){
        std::stringstream strs;
        strs << type_of_filters;
        this->filter = get_filter_bank((char *)strs.str().c_str());
    }

    // Change the norm
    if (this->use_l2_norm)
        this->norm = NORM_L2;

    // Load the non orthogonal filter
    if ((this->type_of_non_orthog_filters > 0) && (this->type_of_non_orthog_filters <= NBR_UNDEC_FILTER))
        this->no_filter = type_undec_filter(this->type_of_non_orthog_filters - 1);

    // Check input parameters
	if ((this->mr_transform != TO_LIFTING) && (this->lift_transform != DEF_LIFT))
        throw std::invalid_argument("-l option is only available with lifting transform.");
	if ((this->mr_transform != TO_UNDECIMATED_MALLAT) && (this->mr_transform != TO_MALLAT) &&
            (this->use_l2_norm))
	    throw std::invalid_argument("-T and -L options are only valid with Mallat transform.");
}

// Destructor
MRTransform::~MRTransform(){
}

// Save transformation method
void MRTransform::Save(MultiResol &mr){

    // Welcome message
    if (this->verbose > 0)
        cout << "  Output path: " << this->m_opath << endl;

    // Check inputs
    if (this->m_opath == "")
        throw std::invalid_argument(
            "Please specify an output image path in 'opath'.");

    // Write the results
    mr.write((char *)this->m_opath.c_str());
}

void MRTransform::Info(){
    // Information message
    cout << "---------" << endl;
    cout << "Information" << endl;
    cout << "Runtime parameters:" << endl;
    cout << "  Number of procs: " << this->nb_procs << endl;
    cout << "  MR transform ID: " << this->type_of_multiresolution_transform << endl;
    cout << "  MR transform name: " << StringTransform(this->mr_transform) << endl;
    cout << "  MR border type: " << this->bord << endl;
    if ((this->mr_transform == TO_MALLAT) || (this->mr_transform == TO_UNDECIMATED_MALLAT)) {
        cout << "  Filter ID: " << this->type_of_filters << endl;
        cout << "  Filter name: " << StringSBFilter(this->filter) << endl;
        if (this->use_l2_norm)
            cout << "   Use L2-norm." << endl;
    }
    if (this->mr_transform == TO_LIFTING) {
        cout << "  Lifting transform ID: " << this->type_of_lifting_transform << endl;
        cout << "  Lifting transform name: " << StringLSTransform(this->lift_transform) << endl;
    }
    cout << "  Number of scales: " << this->number_of_scales << endl;
    if (this->mr_transform == TO_UNDECIMATED_MALLAT)
        cout << "  Number of undecimated scales: " <<  this->nb_of_undecimated_scales << endl;
    if (this->mr_transform == TO_UNDECIMATED_NON_ORTHO) {
        cout << "Undecimated non othogonal filter ID: " << this->type_of_non_orthog_filters << endl;
        cout << "Undecimated non othogonal filter name: " << StringUndecFilter(this->no_filter) << endl;
    }
    if (this->verbose > 1)
        mr.print_info();
    cout << "---------" << endl;
}

// Transform method
py::list MRTransform::Transform(py::array_t<float>& arr, bool save){
    // Load the input image
    /*if (this->verbose > 0)
        cout << "Loading input image '" << this->m_ipath << "'..." << endl;
    Ifloat data;
    io_read_ima_float((char *)this->m_ipath.c_str(), data);
    if (this->verbose > 0)
        cout << "Done." << endl;*/

    // Create the transformation
    Ifloat data = array2image_2d(arr);
    if (!this->mr_initialized) {
        //MultiResol mr;
        //FilterAnaSynt fas;
        //FilterAnaSynt *ptrfas = NULL;
        if ((this->mr_transform == TO_MALLAT) || (this->mr_transform == TO_UNDECIMATED_MALLAT)) {
            fas.Verbose = (Bool)this->verbose;
            fas.alloc(this->filter);
            ptrfas = &fas;
        }
        mr.alloc(data.nl(), data.nc(), this->number_of_scales,
                 this->mr_transform, ptrfas, this->norm,
                 this->nb_of_undecimated_scales, this->no_filter);
        if (this->mr_transform == TO_LIFTING)
            mr.LiftingTrans = this->lift_transform;
        mr.Border = this->bord;
        mr.Verbose = (Bool)this->verbose;
        this->mr_initialized = true;
    }

    // Perform the transformation
    if (this->verbose > 0) {
        cout << "Starting transformation" << endl;
        cout << "Runtime parameters:" << endl;
        cout << "  Number of bands: " << mr.nbr_band() << endl;
        cout << "  Data dimension: " << arr.ndim() << endl;
        cout << "  Array shape: " << arr.shape(0) << ", " << arr.shape(1) << endl;
        cout << "  Save transform: " << save << endl;
    }
    mr.transform(data);

    // Correct the reconstruction error
    if ((this->iter > 1) && (mr.Set_Transform == TRANSF_PYR))
        mr_correct_pyr(data, mr, this->iter);

    // Save transform if requested
    if (save)
        Save(mr);

    // Return the generated bands data
    py::list mr_data;
    for (int s=0; s<mr.nbr_band(); s++) {
        mr_data.append(image2array_2d(mr.band(s)));
    }

    // Get the number of bands for each scale
    py::list mr_scale;
    int nb_bands_count = 0;
    for (int s=0; s<mr.nbr_scale(); s++) {
        nb_bands_count += mr.nbr_band_per_resol(s);
        mr_scale.append(mr.nbr_band_per_resol(s));
    }
    if (nb_bands_count != mr.nbr_band()) {
        mr_scale[py::len(mr_scale) - 1] = 1;
    }

    // Format the result
    py::list mr_result;
    mr_result.append(mr_data);
    mr_result.append(mr_scale);

    return mr_result;
}

// Reconstruction method
py::array_t<float> MRTransform::Reconstruct(py::list mr_data){
    // Welcome message
    if (this->verbose > 0) {
        cout << "Starting Reconstruction" << endl;
        cout << "Runtime parameters:" << endl;
        cout << "  Number of bands: " << py::len(mr_data) << endl;
    }

    // Update transformation
    for (int s=0; s<py::len(mr_data); s++) {
        py::array_t<float> band_array = py::array(mr_data[s]);
        Ifloat band_data = array2image_2d(band_array);
        // cout << "Size of inserted band ";
        // cout << "nb_e:"<< band_data.n_elem() << "/ndim:" << band_data.naxis()\
        // << "/nx:" << band_data.nx() << "/ny:"  << band_data.ny() <<  endl;

        mr.insert_band(band_data, s);
    }

    // Start the reconstruction
    Ifloat data(mr.size_ima_nl(), mr.size_ima_nc(), "Reconstruct");
    mr.recons(data);

    return image2array_2d(data);
}

#endif
