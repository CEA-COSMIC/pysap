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
#include "deconvolve.hpp"
#include "mr_2d1d.hpp"
#include "starlet.hpp"

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
    < int, int, int, int, int, int, int, float, double, double, double, std::string, float, std::string, std::string,
    bool, bool, bool, double, int, int, int, std::string, std::string, int, std::string, 
    bool, bool, bool, int, float, float, float, bool ,bool, bool, bool, bool, std::vector<float>&>(),
        py::arg("type_of_filtering")=(int)(1),
        py::arg("coef_detection_method")=(int)(1),
        py::arg("type_of_multiresolution_transform")=(int)(2),
        py::arg("type_of_filters")=(int)(1),
        py::arg("type_of_non_orthog_filters")=(int)(2),
        py::arg("type_of_noise")=(int)(1),
        py::arg("number_of_scales")=(int)(DEFAULT_NBR_SCALE),
        py::arg("regul_param")=(float)(0.1),
        py::arg("epsilon")=(double)(DEFAULT_EPSILON_FILTERING),
        py::arg("iter_max")=(double)(DEFAULT_MAX_ITER_FILTER),
        py::arg("max_inpainting_iter")=(double)(DEFAULT_MAX_ITER_INPAINTING),
        py::arg("support_file_name")=(std::string)(""),
        py::arg("sigma_noise")=(float)(0.),
        py::arg("flat_image")=(std::string)(""),
        py::arg("rms_map")=(std::string)(""),
        py::arg("missing_data")=(bool)(false),
        py::arg("keep_positiv_sup")=(bool)(false),
        py::arg("write_info_on_prob_map")=(bool)(false),
        py::arg("epsilon_poisson")=(double)(1.00e-03),
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
        py::arg("pas_codeur")=(float)(1),
        py::arg("sigma_gauss")=(float)(0.),
        py::arg("mean_gauss")=(float)(0.),
        py::arg("old_poisson")=(bool)(false),
        py::arg("positiv_ima")=(bool)(DEF_POSITIV_CONSTRAINT),
        py::arg("max_ima")=(bool)(DEF_MAX_CONSTRAINT),
        py::arg("kill_last_scale")=(bool)(false),
        py::arg("write_threshold")=(bool)(false),
        py::arg("tab_n_sigma")=(std::vector<float>&)(v)
    )
    .def("info", &MRFilters::Info)
    .def("filter", &MRFilters::Filter, py::arg("arr"));

  py::class_<MRDeconvolve>(module, "MRDeconvolve")
    .def(py::init<int, int, int, int, float, int, int, float, int, float, bool,
                bool, bool, float, float, float, std::string, std::string,
                std::string, bool, bool, bool, bool, float, float, float>(),
        py::arg("type_of_deconvolution")=(int)(3),
        py::arg("type_of_multiresolution_transform")=(int)(2),
        py::arg("type_of_filters")=(int)(1),
        py::arg("number_of_undecimated_scales")=(int)(-1),
        py::arg("sigma_noise")=(float)(0.),
        py::arg("type_of_noise")=(int)(1),
        py::arg("number_of_scales")=(int)(DEFAULT_NBR_SCALE),
        py::arg("nsigma")=(float)(DEFAULT_N_SIGMA),
        py::arg("number_of_iterations")=(int)(DEFAULT_MAX_ITER_DECONV),
        py::arg("epsilon")=(float)(DEFAULT_EPSILON_DECONV),
        py::arg("psf_max_shift")=(bool)(true),
        py::arg("verbose")=(bool)(false),
        py::arg("optimization")=(bool)(false),
        py::arg("fwhm_param")=(float)(0.),
        py::arg("convergence_param")=(float)(1.),
        py::arg("regul_param")=(float)(0.),
        py::arg("first_guess")=(std::string)(""),
        py::arg("icf_filename")=(std::string)(""),
        py::arg("rms_map")=(std::string)(""),
        py::arg("kill_last_scale")=(bool)(false),
        py::arg("positive_constraint")=(bool)(true),
        py::arg("keep_positiv_sup")=(bool)(false),
        py::arg("sup_isol")=(bool)(false),
        py::arg("pas_codeur")=(float)(1),
        py::arg("sigma_gauss")=(float)(0),
        py::arg("mean_gauss")=(float)(0)
      )
      .def("info", &MRDeconvolve::Info)
      .def("deconvolve", &MRDeconvolve::Deconvolve, py::arg("arr"), py::arg("psf"));

      py::class_<MR2D1D>(module, "MR2D1D")
    .def(py::init<int, bool, bool, int, int >(),
        py::arg("type_of_transform")=(int)(14),
        py::arg("normalize")=(bool)(False),
        py::arg("verbose")=(bool)(False),
        py::arg("NbrScale2d")=(int)(5),
        py::arg("Nbr_Plan")=(int)(4)
      )
    .def("transform", &MR2D1D::Transform, py::arg("Name_Cube_in"))
    .def("reconstruct", &MR2D1D::Reconstruct, py::arg("data"));
    
      py::class_<MRStarlet>(module, "MRStarlet")
    .def(py::init<int, bool, int, int >(),
        py::arg("bord")=(int)(0),
        py::arg("gen2")=(bool)(False),
        py::arg("nb_procs")=(int)(0),
        py::arg("verbose")=(int)(0)
     )
    .def("transform", &MRStarlet::transform, py::arg("arr"), py::arg("ns"))
    .def("recons", &MRStarlet::recons, py::arg("mr_data"),py::arg("adjoint")=(bool)(0))
    .def("info", &MRStarlet::info);
}
