#========================================================#
# Build the CfitsIO dependencies for the project         #
#========================================================#

set(sparse2dVersion 2.1.4)

ExternalProject_Add(sparse2d
    PREFIX sparse2d
    GIT_REPOSITORY https://github.com/CosmoStat/Sparse2D.git
    GIT_TAG v2.1.4
    # GIT_TAG master
    DEPENDS cfitsio
    CONFIGURE_COMMAND cmake ../sparse2d
        -DCMAKE_INSTALL_PREFIX=${SPARSE2D_INSTALL_DIR}/..
        -DCFITSIO_INCLUDE_DIRS=${cfitsio_INCLUDE_DIR}
        -DCFITSIO_LIBRARY_DIRS=${cfitsio_LIBRARY_DIR}
        -DCFITSIO_LIBRARIES=${cfitsio_LIBRARIES}
        -DCMAKE_CXX_FLAGS="-std=c++14"
        -DCMAKE_BUILD_TYPE=RELEASE
    BUILD_COMMAND make install
        -j8
    BUILD_IN_SOURCE 0
    )

set(sparse2d_LIBRARY_DIR ${SPARSE2D_INSTALL_DIR}/../lib/ )
set(sparse2d_INCLUDE_DIR ${SPARSE2D_INSTALL_DIR}/../include/ )
set(sparse2d_LIBRARIES -lmga2d -lsparse3d -lsparse2d -lsparse1d -ltools)
