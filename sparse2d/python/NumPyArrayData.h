/*##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################*/

#pragma once

#include <omp.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <stdexcept>
#include <sstream>
#include <sparse2d/IM_Obj.h>
#include <sparse2d/IM_IO.h>


namespace bp = boost::python;
namespace bn = boost::python::numpy;


#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);


// Helper class for fast access to array elements
template<typename T> class NumPyArrayData
{
    char* m_data;
    const Py_intptr_t* m_strides;

public:
    NumPyArrayData<T>(const bn::ndarray &arr)
    {
        bn::dtype dtype = arr.get_dtype();
        bn::dtype dtype_expected = bn::dtype::get_builtin<T>();

        if (dtype != dtype_expected)
        {
            std::stringstream ss;
            ss << "NumPyArrayData: Unexpected data type (" << bp::extract<const char*>(dtype.attr("__str__")()) << ") received. ";
            ss << "Expected " << bp::extract<const char*>(dtype_expected.attr("__str__")());
            throw std::runtime_error(ss.str().c_str());
        }

        m_data = arr.get_data();
        m_strides = arr.get_strides();
    }

    T* data()
    {
        return reinterpret_cast<T*>(m_data);
    }

    const Py_intptr_t* strides()
    {
        return m_strides;
    }

    // 1D array access
    inline T& operator()(int i)
    {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0]);
    }

    // 2D array access
    inline T& operator()(int i, int j)
    {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1]);
    }

    // 3D array access
    inline T& operator()(int i, int j, int k)
    {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1] + k*m_strides[2]);
    }

    // 4D array access
    inline T& operator()(int i, int j, int k, int l)
    {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1] + k*m_strides[2] + l*m_strides[3]);
    }
};


// Helper function for fast image to array conversion
bn::ndarray image2array_2d(const Ifloat& im){
    // TODO: use buffer
    bn::ndarray arr = bn::zeros(
        bp::make_tuple(im.nl(), im.nc()),
        bn::dtype::get_builtin<double>());
    NumPyArrayData<double> arr_data(arr);
    for (int i=0; i<im.nl(); i++) {
        for (int j=0; j<im.nc(); j++) {
            arr_data(i, j) = im(i, j);
        }
    }
    return arr;
}

Ifloat array2image_2d(const bn::ndarray& arr){
    // Input array checks
    ASSERT_THROW(
        (arr.get_nd() == 2),
        "Expected two-dimensional array");

    // Get the data: force cast to float
    // TODO: use buffer
    Ifloat im(arr.shape(0), arr.shape(1));
    NumPyArrayData<double> arr_data(arr);
    for (int i=0; i<arr.shape(0); i++) {
        for (int j=0; j<arr.shape(1); j++) {
            im(i, j) = (float)arr_data(i, j);
        }
    }
    return im;
}

bn::ndarray image2array_3d(const fltarray& im){
    // TODO: use buffer
    bn::ndarray arr = bn::zeros(
        bp::make_tuple(im.nx(), im.ny(), im.nz()),
        bn::dtype::get_builtin<double>());

    NumPyArrayData<double> arr_data(arr);
    for (int i=0; i<im.nx(); i++) {
        for (int j=0; j<im.ny(); j++) {
            for(int k=0; k<im.nz(); k++){
                arr_data(i, j, k) = im(i, j, k);
            }
        }
    }
    return arr;
}


fltarray array2image_3d(const bn::ndarray& arr){
    // Input array checks
    ASSERT_THROW(
        (arr.get_nd() == 3),
        "Expected three-dimensional array");

    // Get the data: force cast to float
    // TODO: use buffer
    fltarray im(arr.shape(0), arr.shape(1), arr.shape(2));
    NumPyArrayData<double> arr_data(arr);

    for (int i=0; i<arr.shape(0); i++) {
        for (int j=0; j<arr.shape(1); j++) {
            for (int k=0; k<arr.shape(2); k++){
                im(i, j, k) = (float)arr_data(i, j, k);
            }
        }
    }
    return im;
}
