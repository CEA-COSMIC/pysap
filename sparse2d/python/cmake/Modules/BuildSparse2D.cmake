#========================================================#
# Build the CfitsIO dependencies for the project         #
#========================================================#

set(sparse2dVersion 2.0.0)

ExternalProject_Add(sparse2d
    PREFIX sparse2d
    #git GIT_REPOSITORY https://github.com/CosmoStat/Sparse2D.git
    GIT_REPOSITORY https://github.com/AGrigis/Sparse2D.git
    #GIT_TAG v1.0.0
    GIT_TAG master
    DEPENDS cfitsio
    CONFIGURE_COMMAND cmake ../sparse2d
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern
        -DCFITSIO_INCLUDE_DIRS=${cfitsio_INCLUDE_DIR}
        -DCFITSIO_LIBRARY_DIRS=${cfitsio_LIBRARY_DIR}
        -DCFITSIO_LIBRARIES=${cfitsio_LIBRARIES}
        -DCMAKE_BUILD_TYPE=RELEASE
    BUILD_COMMAND make install
        -j8
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 0
    )

set(sparse2d_LIBRARY_DIR ${CMAKE_BINARY_DIR}/extern/lib/ )
set(sparse2d_INCLUDE_DIR ${CMAKE_BINARY_DIR}/extern/include/ )
set(sparse2d_LIBRARIES -lmga2d -lsparse2d -lsparse1d -ltools)

