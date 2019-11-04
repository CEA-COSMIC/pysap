/*##########################################################################
pySAP - Copyright (C) CEA, 2017 - 2018
Distributed under the terms of the CeCILL-B license, as published by
the CEA-CNRS-INRIA. Refer to the LICENSE file or to
http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
for details.
##########################################################################*/

#ifndef NUMPYDATA_H_
#define NUMPYDATA_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <sparse2d/IM_Obj.h>
#include <sparse2d/IM_IO.h>
#include <iostream>

namespace py = pybind11;

// Helper function for fast image to array conversion
py::array_t<float> image2array_2d(const Ifloat &image){

  auto array = py::array_t<float>(image.n_elem());
  auto buffer = array.request();
  float *pointer = (float *) buffer.ptr;

  for (int i=0; i<image.nl(); i++) {
    for (int j=0; j<image.nc(); j++) {
      pointer[j + i * image.nc()] = image(i, j);
    }
  }

  array.resize({image.nl(), image.nc()});

  return array;
}

// Helper function for fast arrat to image conversion
Ifloat array2image_2d(py::array_t<float> &array){

  if (array.ndim() != 2)
    throw std::runtime_error("Input should be 2-D NumPy array");

  auto buffer = array.request();
  float *pointer = (float *) buffer.ptr;

  Ifloat image(array.shape(0), array.shape(1));

  for (int i=0; i<array.shape(0); i++) {
    for (int j=0; j<array.shape(1); j++) {
      image(i, j) = pointer[j + i * array.shape(1)];
    }
  }

  return image;
}

// Helper function for fast image to 3D array conversion
py::array_t<float> image2array_3d(const fltarray &image){

  auto array = py::array_t<float>(image.n_elem());
  auto buffer = array.request();
  float *pointer = (float *) buffer.ptr;

  for (int i=0; i<image.nx(); i++) {
    for (int j=0; j<image.ny(); j++) {
      for(int k=0; k<image.nz(); k++){
        pointer[i + image.nx() * (j + k * image.ny())] = image(i, j, k);
      }
    }
  }

  array.resize({image.nz(), image.ny(), image.nx()});

  return array;
}

// Helper function for fast 3D arrat to image conversion
fltarray array2image_3d(py::array_t<float> &array) {

  if (array.ndim() == 3)
  {
    auto buffer = array.request();
    float *pointer = (float *) buffer.ptr;

    fltarray image(array.shape(2), array.shape(1), array.shape(0));

    for (int i=0; i<array.shape(2); i++) {
      for (int j=0; j<array.shape(1); j++) {
        for (int k=0; k<array.shape(0); k++){
          image(i, j, k) = pointer[i + array.shape(2) * (j + k * array.shape(1))];
        }
      }
    }
    return image;
  }
  else if (array.ndim() == 1)
  {
      fltarray image(len(array));

      auto buffer = array.request();
      float *pointer = (float *) buffer.ptr;

      for (int i = 0; i < len(array); ++i)
          image(i) = pointer[i];
      return image;
  }
  else
    throw std::runtime_error("Input should be 3-D NumPy array");
}

#endif
