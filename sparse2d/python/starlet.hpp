/*##########################################################################
pySAP - Copyright (C) CEA, 2017 - 2018
Distributed under the terms of the CeCILL-B license, as published by
the CEA-CNRS-INRIA. Refer to the LICENSE file or to
http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
for details.
##########################################################################*/

#ifndef STARLET_H_
#define STARLET_H_

// Includes
#include <iostream>
#include <string>
#include <sstream>
#include <sparse2d/IM_Obj.h>
#include <sparse2d/IM_IO.h>
#include <sparse2d/MR_Obj.h>
#include <sparse2d/IM_Prob.h>
#include "numpydata.hpp"


class MRStarlet {

public:
    // Constructor
    MRStarlet(
        int bord=0,
        bool gen2=false,
        int nb_procs=0,
        int verbose=0);

    // Destructor
    ~MRStarlet();
 
    // Information method
    void info();
    
    py::list transform(py::array_t<float>& arr, int Ns=4);

    // Reconstruction method
    py::array_t<float> recons(py::list mr_data, bool adjoint=false);

    // Getter/setter functions for the input/output image path
    void set_opath(std::string path) {this->m_opath = path;}
    string get_opath() const {return m_opath;}

private:
    ATROUS_2D_WT mr;
    bool mr_initialized;
    std::string m_opath;
    int nb_procs;
    int number_of_scales;
    Ifloat *TabTrans;
    bool gen2;
    type_border bord;
    int verbose;
};

// Constructor
MRStarlet::MRStarlet(
        int bord,
        bool gen2,
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
    this->number_of_scales = 0;
    this->verbose = verbose;
    this->mr_initialized = false;
    this->gen2=gen2;
    this->TabTrans=NULL;
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

}

// Destructor
MRStarlet::~MRStarlet(){
}


void MRStarlet::info(){
    // Information message
    cout << "---------" << endl;
    cout << "Information" << endl;
    cout << "Runtime parameters:" << endl;
    cout << "  Number of procs: " << this->nb_procs << endl;
    cout << "  MR border type: " << this->bord << endl;
    cout << "  Number of scales: " << this->number_of_scales << endl;
    if (gen2) cout << "  Second Starlet generation" << endl;
    else  cout << "  First Starlet generation" << endl;
    cout << "---------" << endl;
}

// Transform method
py::list MRStarlet::transform(py::array_t<float>& arr, int Ns){
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
        mr.alloc (TabTrans, data.nl(), data.nc(), Ns);
        mr.Bord = this->bord;
        this->number_of_scales = Ns;
        mr.ModifiedAWT = (Bool)this->gen2;
        this->mr_initialized = true;
    }
    else
    {
        if ((this->number_of_scales != Ns) || (data.nl() != TabTrans[0].nl()) || (data.nc() != TabTrans[0].nc()))
        {
            delete [] TabTrans;
            mr.alloc (TabTrans, data.nl(), data.nc(), Ns);
            mr.Bord = this->bord;
            this->number_of_scales = Ns;
            mr.ModifiedAWT = (Bool)this->gen2;
        }
    }

    // Perform the transformation
    if (this->verbose > 0) {
        cout << "Starting transformation" << endl;
        cout << "Runtime parameters:" << endl;
        if (mr.ModifiedAWT == False) cout << "  First Starlet generation" << endl;
        else cout << "  Second Starlet generation" << endl;
        cout << "  Number of bands: " << this->number_of_scales << endl;
        cout << "  Data dimension: " << arr.ndim() << endl;
        cout << "  Array shape: " << arr.shape(0) << ", " << arr.shape(1) << endl;
    }

    mr.transform(data, TabTrans, Ns, this->nb_procs);

    // Return the generated bands data
            py::list mr_data;
    for (int s=0; s< Ns; s++) {
        mr_data.append(image2array_2d(TabTrans[s]));
    }

    // Format the result
    // py::list mr_result;
    // mr_result.append(mr_data);
    // return mr_result;
    return mr_data;
}
            
// Reconstruction method
py::array_t<float> MRStarlet::recons(py::list mr_data, bool adjoint){
    // Welcome message
    int nx = py::array(mr_data[0]).shape(0);
    int ny = py::array(mr_data[0]).shape(1);
    int ns = py::len(mr_data);
    if (this->verbose > 0) {
        cout << "Starting Reconstruction" << endl;
        cout << "Runtime parameters:" << endl;
        cout << "  Number of bands: " << py::len(mr_data) << endl;
        cout << "  Image size : " << nx << " " << ny  << endl;
    }

    if (!this->mr_initialized) {
        //MultiResol mr;
        mr.alloc (TabTrans, nx, ny, ns);
        mr.Bord = this->bord;
        this->number_of_scales = ns;
        mr.ModifiedAWT = (Bool)this->gen2;
        this->mr_initialized = true;
    }
    
    // Update transformation
    for (int s=0; s<py::len(mr_data); s++) {
        py::array_t<float> band_array = py::array(mr_data[s]);
        Ifloat band_data = array2image_2d(band_array);
        // cout << "Size of inserted band ";
        // cout << "nb_e:"<< band_data.n_elem() << "/ndim:" << band_data.naxis()\
        // << "/nx:" << band_data.nx() << "/ny:"  << band_data.ny() <<  endl;
        TabTrans[s] = band_data;
    }

    // Start the reconstruction
    Ifloat data(TabTrans[0].nl(), TabTrans[0].nc(), "Reconstruct");
    if (adjoint) mr.adjoint_recons(TabTrans, data, this->number_of_scales, this->nb_procs);
    else mr.recons(TabTrans, data, this->number_of_scales, this->nb_procs);
    return image2array_2d(data);
}

#endif
