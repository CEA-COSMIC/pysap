#ifndef MR2D_2D1D_H
#define MR2D_2D1D_H

#include "sparse2d/IM_Obj.h"
#include "sparse2d/IM_IO.h"
#include "sparse2d/SB_Filter.h"
#include "sparse2d/MR1D_Obj.h"
#include "sparse2d/MR_Obj.h"
#include "sparse2d/IM3D_IO.h"
#include "sparse2d/MR3D_Obj.h"
#include "sparse2d/DefFunc.h"

#include "numpydata.hpp"

class MR2D1D
{
    public:
        MR2D1D(int type_of_multiresolution_transform=14, bool normalize=False,
               bool verbose=False, int number_of_scales_2D=5, int number_of_scales=4);
        py::list Transform(py::array_t<float> cube);
        //py::list Reconstruct(py::array_t<float> mr);
        void alloc();
        void Info();
        fltarray get_band(int s2, int s1);
        void transform_per_frame_2D_wt(fltarray data, Ifloat frame);
        void transform_per_frame_1D_wt(fltarray data, fltarray vect);

    private:
        bool normalize;
        bool verbose;
        int number_of_scales_2D;
        int number_of_scales;
        int nx, ny, nz;
        
        int number_band_2D;
        int number_band_1D;
        
        MultiResol wt_2d;
        MR_1D wt_1d;
        fltarray *tab_band;
        sb_type_norm norm;
        type_sb_filter sb_filter;
        type_undec_filter u_filter;
        type_transform  transform;//=TO_MALLAT;
        type_border bord;// = I_MIRROR;
        //float nsigma = 3;
        intarray tab_first_pos_band_nz;
        intarray tab_size_band_nx;
        intarray tab_size_band_ny;
        intarray tab_size_band_nz;
        FilterAnaSynt fas;
};

MR2D1D::MR2D1D(int type_of_multiresolution_transform, bool normalize,
               bool verbose, int number_of_scales_2D, int number_of_scales)
{
    this->normalize = normalize; //(normalize == True) ? False: True; // a checker
    this->verbose = verbose;
    this->number_of_scales_2D = number_of_scales_2D;
    this->number_of_scales = number_of_scales;
    this->norm = NORM_L2;
    this->sb_filter = F_MALLAT_7_9;
    this->bord = I_CONT;
    this->u_filter = DEF_UNDER_FILTER; 

    if ((type_of_multiresolution_transform > 0) && (type_of_multiresolution_transform <= NBR_TRANSFORM+1)) 
        this->transform = (type_transform) (type_of_multiresolution_transform-1);
    
    if ((number_of_scales_2D <= 1) || (number_of_scales_2D > MAX_SCALE_1D))
        throw std::invalid_argument("Error: bad number of scales 2D : " + std::to_string(number_of_scales_2D));
    
    if ((number_of_scales <= 0) || (number_of_scales > MAX_SCALE_1D))
        throw std::invalid_argument("Error: bad number of scales : " + std::to_string(number_of_scales));
}

void MR2D1D::alloc()
{
   FilterAnaSynt *ptrfas = NULL;
    if ((this->transform == TO_MALLAT) || (this->transform == TO_UNDECIMATED_MALLAT))
    {
        this->fas.Verbose = (Bool)this->verbose;
        this->fas.alloc(this->sb_filter);
        ptrfas = &fas;
    }
    //mr 2d init
    int number_undec = -1;                     /*number of undecimated scale */
    if (this->transform == TO_LIFTING)
        wt_2d.LiftingTrans = DEF_LIFT;
    wt_2d.Border = this->bord;
    wt_2d.Verbose = (Bool)this->verbose; 
    wt_2d.alloc (this->ny, this->nx, this->number_of_scales_2D, this->transform,
            ptrfas, this->norm, number_undec, this->u_filter);
    this->number_band_2D = wt_2d.nbr_band();
    wt_2d.ModifiedATWT = True;

    //mr 1d init
    wt_1d.U_Filter = this->u_filter;
    wt_1d.alloc (this->nz, TO1_MALLAT, this->number_of_scales, ptrfas, this->norm, False);   
    this->number_band_1D = wt_1d.nbr_band();
 
    //tab_band init
    this->tab_band = new fltarray [this->number_band_2D];

    this->tab_size_band_nx.resize(this->number_band_2D, this->number_band_1D);
    this->tab_size_band_ny.resize(this->number_band_2D, this->number_band_1D);
    this->tab_size_band_nz.resize(this->number_band_2D, this->number_band_1D);
    
    this->tab_first_pos_band_nz.resize(this->number_band_1D);
    this->tab_first_pos_band_nz(0) = 0;

    for (int b = 0; b < this->number_band_2D; b++) 
    {
        tab_band[b].alloc(wt_2d.size_band_nc(b), wt_2d.size_band_nl(b),  wt_1d.size_ima_np ());
        for (int b1 = 0; b1 < this->number_band_1D; b1++)
        {
            this->tab_size_band_nx(b,b1) = wt_2d.size_band_nc(b);
            this->tab_size_band_ny(b,b1) = wt_2d.size_band_nl(b);
            this->tab_size_band_nz(b,b1) = wt_1d.size_scale_np(b1);
        }
    }
    for (int b1 = 1; b1 < this->number_band_1D; b1++)
        this->tab_first_pos_band_nz(b1) = tab_first_pos_band_nz(b1-1) + tab_size_band_nz(0,b1-1);
}


/*
py::list MR2D1D::Reconstruct(py::array_t<float> multiresol)
{ 
    #if 0
        fltarray data = array2image_3d(multiresol);
        lm_check(LIC_MR3);

        int i,j,b,z;

        if ((data.nx() != this->nx) || (data.ny() != this->ny) || (data.nz() != this->nz))
            data.resize(this->nx, this->ny, this->nz); 

        Ifloat frame(data.ny(), data.nx());
        fltarray vect(data.nz());
        cout << "AVANT 1D?" << std::endl;
        
        // 1D wt 
        if (this->number_band_1D >= 2)
        {
            for (b=0; b < this->number_band_2D; b++)
                for (i=0; i < mr.size_band_nl(b); i++)
                    for (j=0; j < mr.size_band_nc(b); j++) 
                    {
                        for (int b1 = 0; b1 < this->number_band_1D; b1++)
                        for (int p = 0; p < mr_1d.size_scale_np (b1); p++)
                            mr_1d(b1,p) = tab_band[b](j,i,z++); 
                        vect.init();
                        mr_1d.recons(vect);
                        for (int z = 0; z < data.nz(); z++)
                            tab_band[b](j,i,z) = vect(z);
                    }
        }
        mkacout << "APRES1D?" << std::endl;
        // 2D wt 
        for (z = 0; z < data.nz(); z++)
        {
            for (b = 0; b < this->number_band_2D; b++)
            {
                for (i=0; i < mr.size_band_nl(b); i++)
                    for (j=0; j < mr.size_band_nc(b); j++)
                        mr(b,i,j) = tab_band[b](j,i,z);
            }   
            mr.recons(frame);
            for (i=0; i < data.ny(); i++)
                for (j=0; j < data.nx(); j++)
                    data(j,i,z) = frame(i,j);
        }
            //cout << "ETma ICI?" << std::endl;
            py::list mr_data;
            for (int s2 = 0; s2 < this->number_band_2D; s2++) {
                for (int s1 = 0; s1 < this->number_band_1D; s1++) {
                    fltarray tmpband;
                    tmpband = get_band(s2, s1);
                    mr_data.append(image2array_3d(tmpband));
                }
            }
        // cout << "AKRROVE ICI?" << std::endl;
            return mr_data;
    #endif
}
*/

void MR2D1D::transform_per_frame_2D_wt(fltarray data, Ifloat frame)
{
    int i,j,b,z;
    for (z = 0; z < nz; z++)
    {
        for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
                frame(i,j) = data(j,i,z);
        wt_2d.transform(frame);
        for (b=0; b < number_band_2D; b++)
        {
            for (i = 0; i < wt_2d.size_band_nl(b); i++)
            for (j = 0; j < wt_2d.size_band_nc(b); j++)
                this->tab_band[b](j,i,z) = wt_2d(b,i,j);
        }
    }
}

void MR2D1D::transform_per_frame_1D_wt(fltarray data, fltarray vect)
{
    int i,j,b,z;
    for (b = 0; b < number_band_2D; b++)
        for (i = 0; i < wt_2d.size_band_nl(b); i++)
        for (j = 0; j < wt_2d.size_band_nc(b); j++) 
        {
            for (z = 0; z < nz; z++)
                vect(z) = this->tab_band[b](j,i,z);
            wt_1d.transform(vect);
            z = 0;
            for (int b1 = 0; b1 < number_band_1D; b1++)
            {
                for (int p = 0; p < wt_1d.size_scale_np (b1); p++)
                    this->tab_band[b](j,i,z++) = wt_1d(b1,p); 
            }
        }
}

py::list MR2D1D::Transform(py::array_t<float> cube)
{
    int i,j,b,z;

    if (this->verbose)
        Info();

    fltarray data = array2image_3d(cube);

    this->nx = data.nx(); //FIXME: Régler ce problème !!!
    this->ny = data.ny();
    this->nz = data.nz();

    if (this->verbose)
        cout << "Nx = " << nx << " Ny = " << ny << " Nz = " << nz << endl;
    
    if (this->normalize) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
            data(i,j,k) = (data(i,j,k) - data.mean()) / data.sigma();
    }
    
    if (this->verbose)
        cout << "Alloc ...  " << endl;
    alloc();

    //perform the transformation
    if (this->verbose)
        cout << "Transform ...  " << endl;

    Ifloat frame(ny, nx);
    fltarray vect(nz);

    // 2D wt transform per frame
    transform_per_frame_2D_wt(data, frame);

    // 1D wt 
    if (number_band_1D >= 2)
        transform_per_frame_1D_wt(data, vect);

    if (this->verbose)
    {
        for (int s2 = 0; s2 < this->number_band_2D; s2++)
            for (int s1 = 0; s1 < this->number_band_1D; s1++)
            {
                cout << "  Band " << s2 << ", " << s1 << ": " << " Nx = " << tab_size_band_nx(s2,s1);
                cout << ", Ny = " << tab_size_band_ny(s2,s1) <<  ", Nz = " << tab_size_band_nz(s2,s1) << endl;
                fltarray band;
                band = get_band(s2, s1);
                cout << "  Sigma = " << band.sigma() << " Min = " << band.min() << " Max = " << band.max() << endl;
            }
    }

#if 0
    int n_elem = 2;
    for (int s = 0; s < number_band_2D; s++) {
        for (int s1 = 0; s1 < number_band_1D; s1++) 
            n_elem += 3 +  tab_size_band_nx(s,s1)*tab_size_band_ny(s,s1)*tab_size_band_nz(s,s1);
    } 
    fltarray out(n_elem);
    int ind = 2;
    out(0) = number_band_2D;
    out(1) = number_band_1D;
    
    // Ifloat Band;
    for (int s=0; s < number_band_2D; s++)
    for (int s1=0; s1 < number_band_1D; s1++) 
    {
        out(ind++) = tab_size_band_nx(s,s1);
        out(ind++) = tab_size_band_ny(s,s1);
        out(ind++) = tab_size_band_nz(s,s1);
        
        for (int k=0;  k < tab_size_band_nz(s,s1); k++) 
        for (int j=0;  j < tab_size_band_ny(s,s1); j++) 
        for (int i=0;  i < tab_size_band_nx(s,s1); i++)
            out(ind++) = this->tab_band[s] (i, j, k + tab_first_pos_band_nz(s1));
    }
    auto mr_data = image2array_3d(out);
#endif
#if 1
    py::list mr_data;
    for (int s2 = 0; s2 < this->number_band_2D; s2++) {
        for (int s1 = 0; s1 < this->number_band_1D; s1++) {
            fltarray tmpband;
            tmpband = get_band(s2, s1);
            mr_data.append(image2array_3d(tmpband));
        }
    }
#endif
    return mr_data;
    //return image2array_3d(out);
}

fltarray MR2D1D::get_band(int s2, int s1)
{
    int nxb = tab_size_band_nx(s2,s1);
    int nyb = tab_size_band_ny(s2,s1);
    int nzb = tab_size_band_nz(s2,s1);
    fltarray *cube = NULL;
    cube = new fltarray(nxb, nyb, nzb);

    for (int i=0; i < nxb; i++)
    for (int j=0; j < nyb; j++)
    for (int k=0; k < nzb; k++)
        (*cube)(i,j,k) = this->tab_band[s2](i,j,k + tab_first_pos_band_nz(s1));

    return *cube;
}

void MR2D1D::Info()
{
    cout << "Transform = " << StringTransform((type_transform) this->transform) << endl;
    cout << "NbrScale2d = " << this->number_of_scales_2D << endl;
    cout << "NbrScale1d = " << this->number_of_scales << endl;
}

#endif