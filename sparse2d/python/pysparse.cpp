/*##########################################################################
pySAP - Copyright (C) CEA, 2017 - 2018
Distributed under the terms of the CeCILL-B license, as published by
the CEA-CNRS-INRIA. Refer to the LICENSE file or to
http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
for details.
##########################################################################*/

// Includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "transform.hpp"
#include "transform_3D.hpp"

// Defines a python module which will be named "pysparse"
PYBIND11_MODULE(pysparse, module)
{
  module.doc() = "Python bindings for Sparse2D";
  module.attr("__version__") = "0.1.0";
  py::class_<MRTransform>(module, "MRTransform")
    .def(py::init< int, int, int, int, int, bool, int, int, int, int >(),
        py::arg("type_of_multiresolution_transform"),
        py::arg("type_of_lifting_transform")=(int)(3),
        py::arg("number_of_scales")=(int)(4),
        py::arg("iter")=(int)(3),
        py::arg("type_of_filters")=(int)(1),
        py::arg("use_l2_norm")=(bool)(false),
        py::arg("type_of_non_orthog_filters")=(int)(2),
        py::arg("bord")=(int)(0),
        py::arg("nb_procs")=(int)(0),
        py::arg("verbose")=(int)(0)
      )
    .def("info", &MRTransform::Info)
    .def("transform", &MRTransform::Transform, py::arg("arr"), py::arg("save")=(bool)(0))
    .def("reconstruct", &MRTransform::Reconstruct, py::arg("mr_data"))
    .def_property("opath", &MRTransform::get_opath, &MRTransform::set_opath);
  py::class_<MRTransform3D>(module, "MRTransform3D")
    .def(py::init< int, int, int, int, int, bool, int, int >(),
        py::arg("type_of_multiresolution_transform"),
        py::arg("type_of_lifting_transform")=(int)(3),
        py::arg("number_of_scales")=(int)(4),
        py::arg("iter")=(int)(3),
        py::arg("type_of_filters")=(int)(1),
        py::arg("use_l2_norm")=(bool)(false),
        py::arg("nb_procs")=(int)(0),
        py::arg("verbose")=(int)(0)
      )
    .def("info", &MRTransform3D::Info)
    .def("transform", &MRTransform3D::Transform, py::arg("arr"), py::arg("save")=(bool)(0))
    .def("reconstruct", &MRTransform3D::Reconstruct, py::arg("mr_data"))
    .def_property("opath", &MRTransform3D::get_opath, &MRTransform3D::set_opath);

}
