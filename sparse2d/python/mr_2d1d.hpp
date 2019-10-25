#include "sparse2d/IM_Obj.h"
#include "sparse2d/IM_IO.h"
#include "sparse2d/SB_Filter.h"
#include "sparse2d/MR1D_Obj.h"
#include "sparse2d/MR_Obj.h"
#include "sparse2d/IM3D_IO.h"
#include "sparse2d/MR3D_Obj.h"
#include "sparse2d/DefFunc.h"

#include "numpydata.hpp"

class MR2D1D {
  int Nx, Ny, Nz;
  MultiResol *Tab;
  int NbrScale2D;
  int NbrScale1D;
  int NbrBand2D;
  int NbrBand1D;
  MultiResol WT2D;
  MR_1D WT1D;
  fltarray *TabBand;
  intarray TabFirstPosBandNz;
  intarray TabSizeBandNx;
  intarray TabSizeBandNy;
  intarray TabSizeBandNz;
  
  // WT 2D
  sb_type_norm Norm;
  type_sb_filter SB_Filter;
  type_border Bord;
  type_undec_filter U_Filter;
  FilterAnaSynt FAS;

  Bool Verbose;
  Bool Normalize=False;       // normalize data in
  
  public:
        int NbrScale2d;
        int Nbr_Plan;
        type_transform  Transform;
        Bool Reverse;
       MR2D1D (bool reverse=false, int type_of_transform=(int)TO_MALLAT,bool normalize=false,bool verbose=false, int NbrScale2d=5, int Nbr_Plan=4);
       void alloc(int iNx, int iNy, int iNz, type_transform Trans2D, int Ns2D, int Ns1D);
       void perform_transform (fltarray &Data);
       void recons (fltarray &Data);
       int nbr_band_2d () const { return NbrBand2D;}
       int nbr_band_1d () const { return NbrBand1D;}
       int size_band_nx(int s2d, int s1d) const { return TabSizeBandNx(s2d, s1d);}
       int size_band_ny(int s2d, int s1d) const { return TabSizeBandNy(s2d, s1d);}
       int size_band_nz(int s2d, int s1d) const { return TabSizeBandNz(s2d, s1d);}
       float & operator() (int s2, int s1, int i, int j, int k) const;
       fltarray get_band(int s2, int s1);
       py::array_t<float> write();
       py::array_t<float>  transform(py::array_t<float> Name_Cube_In);
    //    ~MR2D1D() { delete [] TabBand;}
};

/****************************************************************************/
MR2D1D::MR2D1D (bool reverse, int type_of_transform,bool normalize,bool verbose, int NbrScale2d, int Nbr_Plan)
{
    Reverse=(Bool)reverse;
    NbrBand2D=NbrBand1D=0;
    Verbose=(Bool)verbose;
    Normalize = (Bool)normalize;
    Bord = I_MIRROR;
    this->NbrScale2d = NbrScale2d;
    this->Nbr_Plan = Nbr_Plan;
    NbrScale2D = NbrScale2d;
    NbrScale1D = Nbr_Plan;

    if ((type_of_transform > 0) && (type_of_transform <= NBR_TRANSFORM+1)) 
        Transform = (type_transform) (type_of_transform-1);
    else
        throw std::invalid_argument("bad type of multiresolution transform: " + std::to_string(type_of_transform));

    if ((NbrScale2d <= 1) || (NbrScale2d > MAX_SCALE_1D)) 
        throw std::invalid_argument("bad number of scales: "+std::to_string(NbrScale2d));

    if ((Nbr_Plan <= 0) || (Nbr_Plan > MAX_SCALE_1D))
        throw std::invalid_argument("bad number of scales: "+std::to_string(Nbr_Plan));
} 

/****************************************************************************/

py::array_t<float>  MR2D1D::write()
{
    int Nelem=2;
    for (int s=0; s < NbrBand2D; s++)
    for (int s1=0; s1 < NbrBand1D; s1++) 
    {
        Nelem += 3 +  TabSizeBandNx(s,s1)*TabSizeBandNy(s,s1)*TabSizeBandNz(s,s1);
    }
    fltarray Data(Nelem);
    int ind=2;
    Data(0) = NbrBand2D;
    Data(1) = NbrBand1D;
    
    // Ifloat Band;
    for (int s=0; s < NbrBand2D; s++)
    for (int s1=0; s1 < NbrBand1D; s1++) 
    {
        Data(ind++) = TabSizeBandNx(s,s1);
        Data(ind++) = TabSizeBandNy(s,s1);
        Data(ind++) = TabSizeBandNz(s,s1);
        
        for (int k=0;  k < TabSizeBandNz(s,s1); k++) 
        for (int j=0;  j < TabSizeBandNy(s,s1); j++) 
        for (int i=0;  i < TabSizeBandNx(s,s1); i++) Data(ind++) = (*this)(s,s1,i,j,k);
  }
  int count = 0;
  py::list result;
  float *buff = Data.buffer();
    for (int i=0; i<Nelem; i++) {
        result.append(buff[i]);
    }
    return result;
}

/****************************************************************************/

float & MR2D1D::operator() (int s2, int s1, int i, int j, int k) const
{
    if ( (i < 0) || (i >= size_band_nx(s2,s1)) ||
            (j < 0) || (j >= size_band_ny(s2,s1)) ||
        (k < 0) || (k >= size_band_nz(s2,s1)) ||
        (s2 < 0) || (s2 >= nbr_band_2d()) ||
        (s1 < 0) || (s1 >= nbr_band_1d()))
    {
        throw std::invalid_argument("Error: invalid number of scales");
    }
  
   return TabBand[s2] (i,j,k+TabFirstPosBandNz(s1));
}

/****************************************************************************/

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
/****************************************************************************/

void MR2D1D::alloc (int iNx, int iNy, int iNz, type_transform Trans2D, int Ns2D, int Ns1D)
{
   Nx = iNx;
   Ny = iNy;
   Nz = iNz;
   NbrScale2D = Ns2D;
   NbrScale1D = Ns1D;
   
   Norm = NORM_L2;
   SB_Filter = F_MALLAT_7_9;
   Bord = I_CONT;
   U_Filter = DEF_UNDER_FILTER; 
   FilterAnaSynt *PtrFAS = NULL;
    if ((Trans2D == TO_MALLAT) || (Trans2D == TO_UNDECIMATED_MALLAT))
    {
        FAS.Verbose = Verbose;
        FAS.alloc(SB_Filter);
        PtrFAS = &FAS;
    }
    int NbrUndec = -1;                     /*number of undecimated scale */
    type_lift LiftingTrans = DEF_LIFT;
    if (Trans2D == TO_LIFTING) WT2D.LiftingTrans = LiftingTrans;
    WT2D.Border = Bord;
    WT2D.Verbose = Verbose;    
    WT2D.alloc (Ny, Nx, Ns2D, Trans2D, PtrFAS, Norm, NbrUndec, U_Filter);
    NbrBand2D = WT2D.nbr_band();
    WT2D.ModifiedATWT = True;
   
    Bool Rebin=False;
    WT1D.U_Filter = U_Filter;
    type_trans_1d Trans1D = TO1_MALLAT;
    WT1D.alloc (Nz, Trans1D, Ns1D, PtrFAS, Norm, Rebin);   
    NbrBand1D = WT1D.nbr_band();
    
   TabBand = new fltarray [NbrBand2D];
   TabSizeBandNx.resize(NbrBand2D, NbrBand1D);
   TabSizeBandNy.resize(NbrBand2D, NbrBand1D);
   TabSizeBandNz.resize(NbrBand2D, NbrBand1D);
   TabFirstPosBandNz.resize(NbrBand1D);
   TabFirstPosBandNz(0) =0;
   for (int b=0; b < NbrBand2D; b++) 
   {
      TabBand[b].alloc(WT2D.size_band_nc(b), WT2D.size_band_nl(b),  WT1D.size_ima_np ());
      for (int b1=0; b1 < NbrBand1D; b1++)
      {
         TabSizeBandNx(b,b1) = WT2D.size_band_nc(b);
	 TabSizeBandNy(b,b1) = WT2D.size_band_nl(b);
	 TabSizeBandNz(b,b1) = WT1D.size_scale_np(b1);
      }
   }
   for (int b1=1; b1 < NbrBand1D; b1++) TabFirstPosBandNz(b1) = TabFirstPosBandNz(b1-1) + TabSizeBandNz(0,b1-1);
}

/****************************************************************************/

void MR2D1D::perform_transform (fltarray &Data)
{
   int i,j,b,z;
   Nx = Data.nx();
   Ny = Data.ny();
   Nz = Data.nz();
   Ifloat Frame(Ny, Nx);
   fltarray Vect(Nz);
   
   // 2D wt transform per frame
   for (z=0; z < Nz; z++)
   {
      for (i=0; i < Ny; i++)
      for (j=0; j < Nx; j++) Frame(i,j) = Data(j,i,z);
      WT2D.transform(Frame);
      for (b=0; b < NbrBand2D; b++)
      {
         for (i=0; i < WT2D.size_band_nl(b); i++)
	 for (j=0; j < WT2D.size_band_nc(b); j++) TabBand[b](j,i,z) = WT2D(b,i,j);
      }
   }
 
   // 1D wt 
   if (NbrBand1D >= 2)
   {
     for (b=0; b < NbrBand2D; b++)
     for (i=0; i < WT2D.size_band_nl(b); i++)
     for (j=0; j < WT2D.size_band_nc(b); j++) 
     {
        for (z=0; z < Nz; z++) Vect(z) = TabBand[b](j,i,z);
        WT1D.transform(Vect);
        z = 0;
        for (int b1=0; b1 < NbrBand1D; b1++)
        {
         for (int p=0; p < WT1D.size_scale_np (b1); p++) TabBand[b](j,i,z++) = WT1D(b1,p); 
        }
      }
   }
}

/****************************************************************************/

void MR2D1D::recons (fltarray &Data)
{
     //    MR2D1D WT;
    //    WT.read(Name_Cube_In);
    //    WT.recons (Dat);
 
    //    if (Verbose == True) cout << "Write result ...  " << endl;
    //    fits_write_fltarr(Name_Out, Dat);
   if ((Data.nx() != Nx) || (Data.ny() != Ny) || (Data.nz() != Nz)) Data.resize(Nx, Ny, Nz); 
   int i,j,b,z;
   Nx = Data.nx();
   Ny = Data.ny();
   Nz = Data.nz();
   Ifloat Frame(Ny, Nx);
   fltarray Vect(Nz);
     
   // 1D wt 
   if (NbrBand1D >= 2)
   {
      for (b=0; b < NbrBand2D; b++)
      for (i=0; i < WT2D.size_band_nl(b); i++)
      for (j=0; j < WT2D.size_band_nc(b); j++) 
      {
         // for (z=0; z < Nz; z++) Vect(z) = TabBand[b](j,i,z);
	z = 0;
        for (int b1=0; b1 < NbrBand1D; b1++)
        for (int p=0; p < WT1D.size_scale_np (b1); p++) WT1D(b1,p) = TabBand[b](j,i,z++); 
         Vect.init();
	 WT1D.recons(Vect);
         for (z=0; z < Nz; z++) TabBand[b](j,i,z) = Vect(z);
     }
   }
   
   // 2D wt 
   for (z=0; z < Nz; z++)
   {
      for (b=0; b < NbrBand2D; b++)
      {
         for (i=0; i < WT2D.size_band_nl(b); i++)
	 for (j=0; j < WT2D.size_band_nc(b); j++) WT2D(b,i,j) = TabBand[b](j,i,z);
      }   
      WT2D.recons(Frame);
      for (i=0; i < Ny; i++)
      for (j=0; j < Nx; j++) Data(j,i,z) = Frame(i,j);
   }
}

/*********************************************************************/
 
py::array_t<float>  MR2D1D::transform(py::array_t<float> Name_Cube_In)
{
    fltarray Dat;
   fitsstruct Header;   
        if (Verbose == True)
        {
           cout << "Transform = " << StringTransform((type_transform) Transform) << endl;
           cout << "NbrScale2d = " << this->NbrScale2d<< endl;
           cout << "NbrScale1d = " << this->Nbr_Plan<< endl;
        }
    
       Dat = array2image_3d(Name_Cube_In);
       int Nx = Dat.nx();
       int Ny = Dat.ny();
       int Nz = Dat.nz();
       if (Verbose == True) cout << "Nx = " << Dat.nx() << " Ny = " << Dat.ny() << " Nz = " << Dat.nz() << endl;
     //Nx = 2 Ny = 3 Nz = 4
       if (Normalize == True)
       {
         double Mean = Dat.mean();
         double Sigma = Dat.sigma();
         for (int i=0;i<Nx;i++)
         for (int j=0;j<Ny;j++)
         for (int k=0;k<Nz;k++) Dat(i,j,k) = (Dat(i,j,k)-Mean)/Sigma;
       }    
    
   
       MR2D1D WT;
       if (Verbose == True) cout << "Alloc ...  " << endl;
       WT.alloc(Nx, Ny, Nz, Transform, NbrScale2d, Nbr_Plan);

       if (Verbose == True) cout << "Transform ...  " << endl;
       WT.perform_transform (Dat);

       if (Verbose == True)cout << "Write result ...  " << endl;
       auto res = WT.write();
              
       if (Verbose == True)
       {
          for (int s2 = 0; s2 < WT.nbr_band_2d (); s2++)
        for (int s1 = 0; s1 < WT.nbr_band_1d (); s1++)
        {
            cout << "  Band " << s2 << ", " << s1 << ": " << " Nx = " << WT.size_band_nx(s2,s1) << ", Ny = " << WT.size_band_ny(s2,s1) <<  ", Nz = " << WT.size_band_nz(s2,s1) << endl;
            fltarray Band;
            Band = WT.get_band(s2, s1);
            cout << "  Sigma = " << Band.sigma() << " Min = " << Band.min() << " Max = " << Band.max() << endl;
        }
       }
       return res;  
}
