////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains an implementation some linear algebra functions for
//!     dense vectors
//!     
//!                    Copyright (C) Simon C Davenport
//!                                                                             
//!     This program is free software: you can redistribute it and/or modify
//!     it under the terms of the GNU General Public License as published by
//!     the Free Software Foundation, either version 3 of the License,
//!     or (at your option) any later version.
//!                                                                             
//!     This program is distributed in the hope that it will be useful, but
//!     WITHOUT ANY WARRANTY; without even the implied warranty of
//!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//!     General Public License for more details.
//!                                                                             
//!     You should have received a copy of the GNU General Public License
//!     along with this program. If not, see <http://www.gnu.org/licenses/>.
//!                                                                             
////////////////////////////////////////////////////////////////////////////////

#ifndef _DENSE_VECTOR_HPP_INCLUDED_
#define _DENSE_VECTOR_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../general/dvec_def.hpp"
#include <random>
#include "blas.hpp"
#if _DEBUG_
#include "../general/debug.hpp"
#endif

namespace utilities
{
    void CopyVector(double* out, const double* in, const int N);
    void ToSubVector(dvec& sub, const dvec& input, const unsigned int offset, 
                     const std::vector<unsigned int>& zeros);
    void ToSubVector(dvec& sub, const dvec& input, const unsigned int offset);
    void FromSubVector(const dvec& sub, dvec& output,  const unsigned int offset, 
                       const std::vector<unsigned int>& zeros);
    void FromSubVector(const dvec& sub, dvec& output, const unsigned int offset);
    void SetToRandomVector(dvec& vec, const double scale, const unsigned int seed);
    void VectorDiff(dvec& output, const dvec& a, const dvec& b);
    void VectorIncrement(dvec& a, const double scale, const dvec& b);
    void VectorSgn(dvec& sgnVec, const dvec& vec);
    void VectorHadamard(dvec& c, const double scale, const dvec& a, const dvec& b);
    void VectorHadamardIncrement(dvec& a, const double scale, const dvec& b);
    double VectorDot(const dvec& a, const dvec& b);
    double VectorL2(const dvec& a);
    double VectorL1(const dvec& a);
    void VectorScale(dvec& output, const double scale, const dvec& input);
}   //  End namespace utilities
#endif
