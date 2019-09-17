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
#include "filter.hpp"

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


  py::class_<MRFilters>(module, "MRFilters")
    .def(py::init
    < int, int, int, int, int, int, int, float, double, double, double, std::string, int, std::string, std::string,
    bool, bool, bool, float, int, int, int, std::string, std::string,
    int, std::string, bool, bool, bool, int, float, float, float, bool ,bool, bool, bool>(),
        py::arg("type_of_filtering")=(int)(1),
        py::arg("coef_detection_method")=(int)(1),
        py::arg("type_of_multiresolution_transform")=(int)(2),
        py::arg("type_of_filters")=(int)(1),
        py::arg("type_of_non_orthog_filters")=(int)(2),
        py::arg("type_of_noise")=(int)(1),
        py::arg("number_of_scales")=(int)(DEFAULT_NBR_SCALE),
        py::arg("regul_param")=(float)(DEFAULT_N_SIGMA),
        py::arg("epsilon")=(double)(DEFAULT_EPSILON_FILTERING),
        py::arg("iter_max")=(double)(DEFAULT_MAX_ITER_FILTER),
        py::arg("max_inpainting_iter")=(double)(DEFAULT_MAX_ITER_INPAINTING),
        py::arg("support_file_name")=(std::string)(""),
        py::arg("sigma_noise")=(int)(0.),
        py::arg("flat_image")=(std::string)(""),
        py::arg("rms_map")=(std::string)(""),
        py::arg("missing_data")=(bool)(false),
        py::arg("keep_positiv_sup")=(bool)(false),
        py::arg("write_info_on_prob_map")=(bool)(false),
        py::arg("epsilon_poisson")=(float)(1.00e-03),
        py::arg("size_block")=(int)(7),
        py::arg("niter_sigma_clip")=(int)(1),
        py::arg("first_scale")=(int)(1),
        py::arg("mask_file_name")=(std::string)(""),
        py::arg("prob_mr_file")=(std::string)(""),
        py::arg("min_event_number")=(int)(0),
        py::arg("background_model_image")=(std::string)(""),
        py::arg("positive_recons_filter") =(bool)(false),
        py::arg("suppress_isolated_pixels")=(bool)(false),
        py::arg("verbose")=(bool)(false),
        py::arg("number_undec")=(int)(-1),
        py::arg("pas_codeur")=(float)(NULL),
        py::arg("sigma_gauss")=(float)(NULL),
        py::arg("mean_gauss")=(float)(NULL),
        py::arg("old_poisson")=(bool)(false),
        py::arg("positiv_ima")=(bool)(DEF_POSITIV_CONSTRAINT),
        py::arg("max_ima")=(bool)(DEF_MAX_CONSTRAINT),
        py::arg("kill_last_scale")=(bool)(false)

    )
    .def("filter", &MRFilters::Filter)
    .def("info", &MRFilters::Info);
}
