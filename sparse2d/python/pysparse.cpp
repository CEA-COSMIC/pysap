/*##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################*/

// Includes
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

// Namespaces
namespace bp = boost::python;
namespace bn = boost::python::numpy;

// Includes
#include "transform.hpp"


// Defines a python module which will be named "pysparse"
BOOST_PYTHON_MODULE(pysparse){

    // Initialize the Numpy support
    Py_Initialize();
    bn::initialize();

    // Declares class MRTransform, specifying the constructor, the input/output
    // path as an attribute visible from python, which can be queried and set 
    // through the C++ get and set functions, and the transform method.
    {
        // Constructor
        typedef bp::class_< MRTransform > MRTransform_exposer_t;
        MRTransform_exposer_t MRTransform_exposer = MRTransform_exposer_t(
            "MRTransform",
            bp::init< int, bp::optional< int, int, int, int, bool, int, int, int > >(
                ( bp::arg("type_of_multiresolution_transform"),
                  bp::arg("type_of_lifting_transform")=(int)(3),
                  bp::arg("number_of_scales")=(int)(4),
                  bp::arg("iter")=(int)(3),
                  bp::arg("type_of_filters")=(int)(1),
                  bp::arg("use_l2_norm")=(bool)(false),
                  bp::arg("type_of_non_orthog_filters")=(int)(2),
                  bp::arg("nb_procs")=(int)(0),
                  bp::arg("verbose")=(int)(0) )
            )
        );
        bp::scope MRTransform_scope( MRTransform_exposer );
        bp::implicitly_convertible< int, MRTransform >();

        // Information method
        {
            typedef void ( ::MRTransform::*Info_function_type)( ) ;
            MRTransform_exposer.def( 
                "info",
                Info_function_type( &::MRTransform::Info )
            );
        }

        // Transform method
        {
            typedef ::bp::list ( ::MRTransform::*Transform_function_type)( ::bn::ndarray, bool ) ;
            MRTransform_exposer.def( 
                "transform",
                Transform_function_type( &::MRTransform::Transform ),
                ( bp::arg("arr"), bp::arg("save")=(bool)(0) )
            );
        }

        // Reconstruction method
        {
            typedef ::bn::ndarray ( ::MRTransform::*Reconstruct_function_type)( bp::list ) ;
            MRTransform_exposer.def( 
                "reconstruct",
                Reconstruct_function_type( &::MRTransform::Reconstruct ) );
        }

        // Output path accessors
        {
            typedef ::std::string ( ::MRTransform::*get_opath_function_type)(  ) const;
            typedef void ( ::MRTransform::*set_opath_function_type)( ::std::string ) ;
            MRTransform_exposer.add_property( 
                "opath",
                get_opath_function_type( &::MRTransform::get_opath ),
                set_opath_function_type( &::MRTransform::set_opath ) );  
        }

    }

    // Module property
    bp::scope().attr("__version__") = "0.0.1";
    bp::scope().attr("__doc__") = "Python bindings for ISAP";

}
