##  This file generates the configure script for the entanglement spectrum program suite

# Process this file with autoconf to produce a configure script.

AC_INIT(Quantum State Neutral Networks, 1.0, simon.davenport2@gmail.com)
AC_CONFIG_SRCDIR(src/)
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE([subdir-objects])

# Option to use the mkl library
AC_ARG_ENABLE([intel_mkl],
  [AS_HELP_STRING([--enable-intel-mkl], [Set to yes to use intel mkl libray for cache efficient Lapack routines, if available])])

# Extra arguments to configure boost program options
AC_ARG_WITH([boost_include],
            [AC_HELP_STRING([--with-boost-include=DIR],
                            [boost program options include directory])],
            [CPPFLAGS="-I$withval $CPPFLAGS"])
            
AC_ARG_WITH([boost_lib],
            [AC_HELP_STRING([--with-boost-lib=DIR],
                            [boost program options lib directory])],
            [LDFLAGS="-L$withval $LDFLAGS"])

# Extra arguments to configure lapack
AC_ARG_WITH([lapack_lib],
            [AC_HELP_STRING([--with-lapack-lib=DIR],
                            [lapack lib directory])],
            [LDFLAGS="-L$withval $LDFLAGS"])

# Extra arguments to configure blas
AC_ARG_WITH([blas_lib],
            [AC_HELP_STRING([--with-blas-lib=DIR],
                            [blas lib directory])],
            [LDFLAGS="-L$withval $LDFLAGS"])

# Remove the default CXXFLAGS.
: ${CXXFLAGS=""}

# Add check for c++11 availability and set other compiler flags
CPPFLAGS="$CPPFLAGS -std=c++11 -O3  -Wall -Wfatal-errors"

# Check for a MPI c++ compiler, preferably one from intel
AC_PROG_CXX([mpicpc mpicxx mpicxxintel mpg++])

# Check for an intel fortran compiler, to be used for linking to intel mkl
# libraries only.

AS_IF([test "x$enable_intel_mkl" = "xyes"],[AC_PROG_FC([mpif90intel mpif90 mpifort])])

AS_IF([test "x$enable_intel_mkl" != "xyes"],[FC="$CXX"])

# Set the language to C++ in preparation for the further checks
AC_LANG(C++)

# Check for boost program options header
AC_CHECK_HEADERS(boost/program_options.hpp)

# Add fortran compiler libraries to the LIB variable
AS_IF([test "x$enable_intel_mkl" = "xyes"],[LIBS="$LIBS -mkl -nofor_main -cxxlib"])

#   Add common library flags
LIBS="$LIBS -lblas -llapack -lm -lboost_program_options"

# Automatically generate Makefiles in each directory level
AC_OUTPUT(Makefile src/Makefile)
