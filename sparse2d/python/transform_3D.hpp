/*##########################################################################
pySAP - Copyright (C) CEA, 2017 - 2018
Distributed under the terms of the CeCILL-B license, as published by
the CEA-CNRS-INRIA. Refer to the LICENSE file or to
http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
for details.
##########################################################################*/

#ifndef TRANSFORM_3D_H_
#define TRANSFORM_3D_H_

/*Availables transforms:
1: Mallat 3D
2: Lifting
3: A trous*/

// Includes
#include <iostream>
#include <string>
#include <sstream>
#include <typeinfo>
#include <sparse2d/IM_Obj.h>
#include <sparse2d/IM_IO.h>
#include <sparse2d/IM3D_IO.h>
#include <sparse2d/MR3D_Obj.h>
#include <sparse2d/MR_Obj.h>
#include <sparse2d/IM_Prob.h>
#include "numpydata.hpp"



#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

class MRTransform3D {

public:
    // Constructor
    MRTransform3D(
        int type_of_multiresolution_transform,
        int type_of_lifting_transform=3,
        int number_of_scales=4,
        int iter=3,
        int type_of_filters=1,
        bool use_l2_norm=false,
        int nb_procs=0,
        int verbose=0);

    // Destructor
    ~MRTransform3D();

    // Save transformation method
    void Save(MR_3D &mr);

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
    MR_3D mr;
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
    int nb_procs;
    int verbose;

    type_trans_3d mr_transform = TO3_MALLAT;
    type_lift lift_transform = DEF_LIFT;
    type_sb_filter filter = F_MALLAT_7_9;

    sb_type_norm norm = NORM_L1;
};

// Constructor
MRTransform3D::MRTransform3D(
        int type_of_multiresolution_transform,
        int type_of_lifting_transform,
        int number_of_scales,
        int iter,
        int type_of_filters,
        bool use_l2_norm,
        int nb_procs,
        int verbose){
    // Define instance attributes
    this->type_of_multiresolution_transform = type_of_multiresolution_transform;
    this->type_of_lifting_transform = type_of_lifting_transform;
    this->number_of_scales = number_of_scales;
    this->iter = iter;
    this->type_of_filters = type_of_filters;
    this->use_l2_norm = use_l2_norm;
    this->verbose = verbose;
    this->mr_initialized = false;
    bool use_filter = false;
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
    if ((this->type_of_multiresolution_transform > 0) && (this->type_of_multiresolution_transform <= NBR_TRANS_3D+1))
        this->mr_transform = type_trans_3d(this->type_of_multiresolution_transform - 1);
    else
        throw std::invalid_argument("Invalid MR transform number.");

    // Load the lifting transform
    if ((this->type_of_lifting_transform > 0) && (this->type_of_lifting_transform <= NBR_LIFT))
        this->lift_transform = type_lift(this->type_of_lifting_transform);
    else
        throw std::invalid_argument("Invalid lifting transform number.");

    // Check the number of scales
    if ((this->number_of_scales <= 1) || (this->number_of_scales > MAX_SCALE_1D))
        throw std::invalid_argument("Bad number of scales ]1; MAX_SCALE].");

    // Check the number of iterations
    if ((this->iter <= 1) || (this->iter > 20))
        throw std::invalid_argument("Bad number of iteration ]1; 20].");

    // Load the filter
    if (this->type_of_filters != 1){
        std::stringstream strs;
        strs << type_of_filters;
        this->filter = get_filter_bank((char *)strs.str().c_str());
        use_filter = true;
    }

    // Change the norm
    if (this->use_l2_norm)
        this->norm = NORM_L2;

    // Check compatibility between parameters
	if ((this->mr_transform != TO3_MALLAT) && (this->use_l2_norm || use_filter))
        throw std::invalid_argument("transforms other than Mallat should not be used with filters and L2 norm");
	if ((this->mr_transform != TO3_LIFTING) && (this->lift_transform != DEF_LIFT))
	    throw std::invalid_argument("Non lifting transforms can only be used with integer Haar WT as lifting scheme:");

}

// Destructor
MRTransform3D::~MRTransform3D(){
}

// Save transformation method
void MRTransform3D::Save(MR_3D &mr){

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

void MRTransform3D::Info(){
    // Information message
    cout << "---------" << endl;
    cout << "Information" << endl;
    cout << "Runtime parameters:" << endl;
    cout << "  Number of procs: " << this->nb_procs << endl;
    cout << "  MR transform ID: " << this->type_of_multiresolution_transform << endl;
    cout << "  MR transform name: " << StringTransf3D(this->mr_transform) << endl;
    if ((this->mr_transform == TO3_MALLAT)) {
        cout << "  Filter ID: " << this->type_of_filters << endl;
        cout << "  Filter name: " << StringSBFilter(this->filter) << endl;
        if (this->use_l2_norm)
            cout << "   Use L2-norm." << endl;
        }
    if (this->mr_transform == TO3_LIFTING) {
        cout << "  Lifting transform ID: " << this->type_of_lifting_transform << endl;
        cout << "  Lifting transform name: " << StringLSTransform(this->lift_transform) << endl;
        }
    cout << "  Number of scales: " << this->number_of_scales << endl;
    cout << "---------" << endl;
    }


// Transform method
py::list MRTransform3D::Transform(py::array_t<float>& arr, bool save){

    // Create the transformation
    fltarray data = array2image_3d(arr);
    if (!this->mr_initialized) {
        if ((this->mr_transform == TO3_MALLAT)) {
            fas.Verbose = (Bool)this->verbose;
            fas.alloc(this->filter);
            ptrfas = &fas;
        }

        mr.alloc(data.nx(), data.ny(), data.nz(), this->mr_transform,
                 this->number_of_scales, ptrfas, this->norm);

        if (this->mr_transform == TO3_LIFTING)
            mr.LiftingTrans = this->lift_transform;

        mr.Verbose = (Bool)this->verbose;
        this->mr_initialized = true;
    }

    // Perform the transformation
    if (this->verbose > 0) {
        cout << "Starting transformation" << endl;
        cout << "Runtime parameters:" << endl;
        cout << "  Number of bands: " << mr.nbr_band() << endl;
        cout << "  Data dimension: " << arr.ndim() << endl;
        cout << "  Array shape: " << arr.shape(0) << ", " << arr.shape(1) << ", " << arr.shape(2) << endl;
        cout << "  Save transform: " << save << endl;
    }

    ASSERT_THROW(
        ((int)pow(2, this->number_of_scales) <= (int)min(arr.shape(0), min(arr.shape(1), arr.shape(2)))),
        "Number of scales is too damn high (for the size of the data)");

    mr.transform(data);

    // Save transform if requested
    if (save)
        Save(mr);

    // Return the generated bands data
    py::list mr_data;
    for (int s=0; s<mr.nbr_band(); s++) {
        fltarray tmpband;
        mr.get_band(s, tmpband);
        mr_data.append(image2array_3d(tmpband));
    }

    // Get the number of bands for each scale
    py::list mr_scale;

    // WARNING: This code is a fix as the method nbr_band_per_resol() hasn't
    // been implemented in the 3D case

    // For decimated transforms Mallat and Lifting, this is constant
    int nbr_band_per_resol_cst = 7;
    // Idem for undecimated Atrous transform
    if(this->mr_transform == TO3_ATROUS ){nbr_band_per_resol_cst = 1;}

    int nb_bands_count = 0;
    for (int s=0; s<mr.nbr_scale(); s++) {
        nb_bands_count += nbr_band_per_resol_cst;
        mr_scale.append(nbr_band_per_resol_cst);
    }
    if (nb_bands_count != mr.nbr_band()) {
        mr_scale[py::len(mr_scale) - 1] = 1;
    }

    // Format the result
    py::list mr_result;
    mr_result.append(mr_data);
    mr_result.append(mr_scale);

    // cout << "mr_result[1]: " << py::extract<std::string>(py::object(mr_result[1]).attr("__str__")())() << endl;

    return mr_result;
}

// Reconstruction method
py::array_t<float> MRTransform3D::Reconstruct(py::list mr_data){
    // Welcome message
    if (this->verbose > 0) {
        cout << "Starting Reconstruction" << endl;
        cout << "Runtime parameters:" << endl;
        cout << "  Number of bands: " << py::len(mr_data) << endl;
    }

    // Update transformation
    for (int s=0; s<py::len(mr_data); s++) {
        py::array_t<float> band_array = py::array(mr_data[s]);
        fltarray band_data = array2image_3d(band_array);
        // cout << "Size of inserted band ";
        // cout << "nb_e:"<< band_data.n_elem() << "/ndim:" << band_data.naxis()\
        // << "/nx:" << band_data.nx() << "/ny:"  << band_data.ny() << "nz:"  << band_data.nz() <<  endl;

        mr.insert_band(s, band_data);
    }

    int Nx = mr.size_cube_nx();
    int Ny = mr.size_cube_ny();
    int Nz = mr.size_cube_nz();

    // Start the reconstruction
    fltarray data(Nx, Ny, Nz, "Reconstruct");
    mr.recons(data);

    return image2array_3d(data);
}

#endif
