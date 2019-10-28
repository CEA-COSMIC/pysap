#include "sparse2d/IM_Obj.h"
#include "sparse2d/IM_IO.h"
#include "sparse2d/MR1D_Obj.h"
#include "sparse2d/MR_Obj.h"
#include "sparse2d/IM3D_IO.h"
#include "sparse2d/MR3D_Obj.h"
#include "sparse2d/DefFunc.h"

#include "numpydata.hpp"

class MR2D1D {

    public:
        MR2D1D (int type_of_transform=(int)TO_MALLAT,bool normalize=false,bool verbose=false, int nb_scale_2d=5, int nb_scale_1d=4);

        void alloc();
        void Info();
        void perform_transform (fltarray &Data);
        fltarray write_result(int Nelem);

        float & operator() (int s2, int s1, int i, int j, int k) const;
        fltarray get_band(int s2, int s1);

        py::array_t<float> Reconstruct(py::array_t<float> data);
        py::array_t<float> Transform(py::array_t<float> Name_Cube_In);
    
    private:
        int Nx, Ny, Nz;
        int nb_scale_2d;
        int nb_scale_1d;
        int nbr_band_2d;
        int nbr_band_1d;
        type_transform  transform;
        Bool verbose;
        bool normalize=False;

        MultiResol WT2D;
        MR_1D WT1D;
        fltarray *TabBand;
        intarray tab_first_pos_band_nz;
        intarray size_band_nx;
        intarray size_band_ny;
        intarray size_band_nz;
        FilterAnaSynt fas;
};

MR2D1D::MR2D1D (int type_of_transform,bool normalize,bool verbose, int nb_scale_2d, int nb_scale_1d)
{
    nbr_band_2d = nbr_band_1d = 0;
    this->verbose = (Bool)verbose;
    this->normalize = normalize;
    this->nb_scale_2d = nb_scale_2d;
    this->nb_scale_1d = nb_scale_1d;


    if ((type_of_transform > 0) && (type_of_transform <= NBR_TRANSFORM+1)) 
        transform = (type_transform) (type_of_transform-1);
    else
        throw std::invalid_argument("bad type of multiresolution transform: " + std::to_string(type_of_transform));

    if ((nb_scale_2d <= 1) || (nb_scale_2d > MAX_SCALE_1D)) 
        throw std::invalid_argument("bad number of scales: "+std::to_string(nb_scale_2d));

    if ((nb_scale_1d <= 0) || (nb_scale_1d > MAX_SCALE_1D))
        throw std::invalid_argument("bad number of scales: "+std::to_string(nb_scale_1d));
} 


float & MR2D1D::operator() (int s2, int s1, int i, int j, int k) const
{
    if ( (i < 0) || (i >= size_band_nx(s2,s1)) ||
            (j < 0) || (j >= size_band_ny(s2,s1)) ||
        (k < 0) || (k >= size_band_nz(s2,s1)) ||
        (s2 < 0) || (s2 >= nbr_band_2d) ||
        (s1 < 0) || (s1 >= nbr_band_1d))
    {
        throw std::invalid_argument("Error: invalid number of scales");
    }
  
   return TabBand[s2] (i,j,k+tab_first_pos_band_nz(s1));
}

fltarray MR2D1D::get_band(int s2, int s1)
{
    int Nxb = size_band_nx(s2,s1);
    int Nyb = size_band_ny(s2,s1);
    int Nzb = size_band_nz(s2,s1);
    fltarray *Cube_Return = NULL;

    Cube_Return = new fltarray(Nxb, Nyb, Nzb);
    for (int i=0; i < Nxb; i++)
    for (int j=0; j < Nyb; j++)
    for (int k=0; k < Nzb; k++) (*Cube_Return)(i,j,k) = (*this)(s2,s1,i,j,k);
    
    return (*Cube_Return);
}

void MR2D1D::alloc()
{
    FilterAnaSynt *ptrfas = NULL;
    if ((transform == TO_MALLAT) || (transform == TO_UNDECIMATED_MALLAT))
    {
        fas.Verbose = verbose;
        fas.alloc(F_MALLAT_7_9); //sb_filter
        ptrfas = &fas;
    }
    if (transform == TO_LIFTING)
        WT2D.LiftingTrans = DEF_LIFT;
    WT2D.Border = I_CONT;
    WT2D.Verbose = verbose;    
    WT2D.alloc (Ny, Nx, nb_scale_2d, transform, ptrfas, NORM_L2, -1, DEF_UNDER_FILTER);
    nbr_band_2d = WT2D.nbr_band();
    WT2D.ModifiedATWT = True;

    WT1D.U_Filter = DEF_UNDER_FILTER;
    WT1D.alloc (Nz, TO1_MALLAT, nb_scale_1d, ptrfas, NORM_L2, False);   
    nbr_band_1d = WT1D.nbr_band();

    TabBand = new fltarray [nbr_band_2d];
    size_band_nx.resize(nbr_band_2d, nbr_band_1d);
    size_band_ny.resize(nbr_band_2d, nbr_band_1d);
    size_band_nz.resize(nbr_band_2d, nbr_band_1d);
    tab_first_pos_band_nz.resize(nbr_band_1d);
    tab_first_pos_band_nz(0) =0;

    for (int b=0; b < nbr_band_2d; b++) 
    {
        TabBand[b].alloc(WT2D.size_band_nc(b), WT2D.size_band_nl(b),  WT1D.size_ima_np ());
        for (int b1=0; b1 < nbr_band_1d; b1++)
        {
            size_band_nx(b,b1) = WT2D.size_band_nc(b);
            size_band_ny(b,b1) = WT2D.size_band_nl(b);
            size_band_nz(b,b1) = WT1D.size_scale_np(b1);
        }
    }
    for (int b1=1; b1 < nbr_band_1d; b1++)
        tab_first_pos_band_nz(b1) = tab_first_pos_band_nz(b1-1) + size_band_nz(0,b1-1);
}

void MR2D1D::perform_transform (fltarray &Data)
{
    Nx = Data.nx();
    Ny = Data.ny();
    Nz = Data.nz();

    Ifloat Frame(Ny, Nx);
    fltarray Vect(Nz);

    // 2D wt transform per frame
    for (int z = 0; z < Nz; z++)
    {
        for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            Frame(i,j) = Data(j,i,z);
        WT2D.transform(Frame);
        for (int b = 0; b < nbr_band_2d; b++)
        {
            for (int i = 0; i < WT2D.size_band_nl(b); i++)
            for (int j = 0; j < WT2D.size_band_nc(b); j++)
                TabBand[b](j,i,z) = WT2D(b,i,j);
        }
    }

    // 1D wt
    if (nbr_band_1d >= 2)
    {
        int z =0;
        for (int b = 0; b < nbr_band_2d; b++)
        for (int i = 0; i < WT2D.size_band_nl(b); i++)
        for (int j = 0; j < WT2D.size_band_nc(b); j++) 
        {
            for (int z = 0; z < Nz; z++)
                Vect(z) = TabBand[b](j,i,z);
            WT1D.transform(Vect);
            z = 0;
            for (int b1 = 0; b1 < nbr_band_1d; b1++)
            for (int p = 0; p < WT1D.size_scale_np (b1); p++)
                TabBand[b](j,i,z++) = WT1D(b1,p);
        }
    }
}

py::array_t<float> MR2D1D::Reconstruct(py::array_t<float> data)
{
    fltarray Tab = array2image_3d(data);

    int ind=2;
    for (int s=0; s <nbr_band_2d; s++) 
    for (int s1=0; s1 < nbr_band_1d; s1++)
    {
       int Nxb = (int) Tab(ind++);
       int Nyb = (int) Tab(ind++);
       int Nzb = (int) Tab(ind++);
       for (int k=0;  k < Nzb; k++) 
       for (int j=0;  j < Nyb; j++)
       for (int i=0;  i < Nxb; i++)
            (*this)(s,s1,i,j,k) = Tab(ind++);
    }

    fltarray Data(Nx, Ny, Nz);
    Ifloat Frame(Ny, Nx);
    fltarray Vect(Nz);

    // 1D wt 
    if (nbr_band_1d >= 2)
    {
        int z = 0;
        for (int b = 0; b < nbr_band_2d; b++)
        for (int i = 0; i < WT2D.size_band_nl(b); i++)
        for (int j = 0; j < WT2D.size_band_nc(b); j++) 
        {
            z = 0;
            for (int b1=0; b1 < nbr_band_1d; b1++)
            for (int p=0; p < WT1D.size_scale_np (b1); p++)
                WT1D(b1,p) = TabBand[b](j,i,z++); 
            Vect.init();
            WT1D.recons(Vect);
            for (int z=0; z < Nz; z++)
                TabBand[b](j,i,z) = Vect(z);
        }
    }
    // 2D wt 
    for (int z=0; z < Nz; z++)
    {
        for (int b=0; b < nbr_band_2d; b++)
        {
            for (int i=0; i < WT2D.size_band_nl(b); i++)
            for (int j=0; j < WT2D.size_band_nc(b); j++)
                WT2D(b,i,j) = TabBand[b](j,i,z);
        }   
        WT2D.recons(Frame);
        for (int i=0; i < Ny; i++)
        for (int j=0; j < Nx; j++)
            Data(j,i,z) = Frame(i,j);
    }
    return image2array_3d(Data);
}
 
py::array_t<float> MR2D1D::Transform(py::array_t<float> Name_Cube_In)
{
    fltarray Dat = array2image_3d(Name_Cube_In);

    Nx = Dat.nx();
    Ny = Dat.ny();
    Nz = Dat.nz();
        
    if (normalize)
    {
        double Mean = Dat.mean();
        double Sigma = Dat.sigma();
        for (int i = 0;i<Nx;i++)
        for (int j = 0;j<Ny;j++)
        for (int k = 0;k<Nz;k++)
            Dat(i,j,k) = (Dat(i,j,k) - Mean) / Sigma;
    }    

    alloc();

    perform_transform(Dat);
            
    if (verbose)
        Info();

    int Nelem=2;
    for (int s=0; s < nbr_band_2d; s++)
    for (int s1=0; s1 < nbr_band_1d; s1++) 
        Nelem += 3 +  size_band_nx(s,s1)*size_band_ny(s,s1)*size_band_nz(s,s1);
    fltarray Data = write_result(Nelem);

    py::list result;
    float *buff = Data.buffer();
    for (int i=0; i<Nelem; i++) {
        result.append(buff[i]);
    }
    return result;
}

fltarray MR2D1D::write_result(int Nelem)
{
    fltarray Data(Nelem);
    int ind=2;
    Data(0) = nbr_band_2d;
    Data(1) = nbr_band_1d;
    
    // Ifloat Band;
    for (int s=0; s < nbr_band_2d; s++)
    for (int s1=0; s1 < nbr_band_1d; s1++) 
    {
        Data(ind++) = size_band_nx(s,s1);
        Data(ind++) = size_band_ny(s,s1);
        Data(ind++) = size_band_nz(s,s1);
        
        for (int k=0;  k < size_band_nz(s,s1); k++) 
        for (int j=0;  j < size_band_ny(s,s1); j++) 
        for (int i=0;  i < size_band_nx(s,s1); i++)
            Data(ind++) = (*this)(s,s1,i,j,k);
    }
    return Data;
}

void MR2D1D::Info()
{
    cout << "Transform = " << StringTransform((type_transform) transform) << endl;
    cout << "nb_scale_2d = " << this->nb_scale_2d<< endl;
    cout << "NbrScale1d = " << this->nb_scale_1d<< endl;
    cout << "Nx = " << Nx << " Ny = " << Ny << " Nz = " << Nz << endl;
    cout << endl;
    for (int s2 = 0; s2 < nbr_band_2d; s2++)
        for (int s1 = 0; s1 < nbr_band_1d; s1++)
        {
            cout << "  Band " << s2 << ", " << s1 << ": " << " Nx = " << size_band_nx(s2,s1) << ", Ny = " << size_band_ny(s2,s1) <<  ", Nz = " << size_band_nz(s2,s1) << endl;
            fltarray Band;
            Band = get_band(s2, s1);
            cout << "  Sigma = " << Band.sigma() << " Min = " << Band.min() << " Max = " << Band.max() << endl;
        }
}
