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

#include "dense_vector.hpp"

namespace utilities
{
    //!
    //! Copy part of a vector
    //!
    void CopyVector(double* out, const double* in, unsigned int N)
    {
        ccopy_(&N, &in, &one, &out, &one);
    }

    //!
    //! Extract elements of the input to be written to a sub vector. Zero values in
    //! the sub vector are skipped.
    //!
    void ToSubVector(
        dvec& sub, 
        const dvec& input, 
        const unsigned int offset, 
        const std::vector<unsigned int>& zeros)
    {
        double* p_sub = sub.data();
        double* p_input = input.data()+offset;
        unsigned int nnzConsecutive = 0;
        for(auto it_zeros = zeros.begin(); it_zeros < zeros.end(); ++it_zeros, ++p_sub)
        {
            nnzConsecutive = *it_zeros - nnzConsecutive;
            CopyVector(p_sub, p_input, runSize);
        }
        nnzConsecutive = sub.size() - nnzConsecutive;
        CopyVector(p_sub, p_input, nnzConsecutive);
    }

    //!
    //! Overload for ToSubVector where non zeros are present
    //!
    void ToSubVector(
        dvec& sub, 
        const dvec& input, 
        const unsigned int offset)
    {
        CopyVector(pub.data(), input.data()+offset, sub.size());
    }
    
    //!
    //! Extract elements of the sub-vector to be written to the output. Zero values in
    //! the sub-vector are skipped.
    //!
    void FromSubVector(
        const dvec& sub, 
        dvec& output, 
        const unsigned int offset,
        const std::vector<unsigned int>& zeros)
    {
        double* p_sub = sub.data();
        double* p_output = output.data()+offset;
        unsigned int runSize = 0;
        for(auto it_zeros = zeros.begin(); it_zeros < zeros.end(); ++it_zeros, ++p_sub)
        {
            runSize = *it_zeros - runSize;
            CopyVector(p_output, p_sub, runSize);
        }
        runSize = sub.size() - runSize;
        CopyVector(p_output, p_sub, runSize);
    }
    
    //!
    //! Overload for FromSubVector where non zeros are present
    //!
    void FromSubVector(
        const dvec& sub, 
        dvec& output, 
        const unsigned int offset)
    {
        CopyVector(output.data()+offset, sub.data(), sub.size());
    }    
    
    //!
    //! Set to a random vector using the given seed
    //!
    void SetToRandomVector(
        std::vector<T>& vec, 
        const double scale, 
        const unsigned int seed)
    {
        std::minstd_rand generator;
        std::uniform_real_distribution<T> distribution(-scale, scale);
        generator.seed(seed);
        std::for_each(vec.begin(), vec.end(), std::bind(distribution, generator));
    }
    
    //!
    //! Compute output = a - b
    //!
    void VectorDiff(
        dvec& output, 
        const dvec& a, 
        const dvec& b)
    {
        static double mOne = -1;
        int N = output.size();
        output = a;
        daxpy_(&N, &mOne, b.data(), &one, output.data(), &one);
    }
    
    //!
    //! Compute a := a + scale*b
    //!
    void VectorIncrement(
        dvec& a, 
        const double scale,
        const dvec& b)
    {
        int N = output.size();
        daxpy_(&N, &scale, b.data(), &one, a.data(), &one);
    }
    
    //!
    //! Compute: output_i = scale*a_i b_i
    //!
    void VectorHadamard(
        dvec& output, 
        const double scale,
        const dvec& a, 
        const dvec& b)
    {
        int N = output.size();
        static const int K = 0;
        static const double BETA = 0.0;
        dsbmv_("L", &N, &K, &scale, a.data(), &one, b.data(), 
               &one, &BETA, output.data(), &one);
    }

    //!
    //! Compute the dot product: output = sum_i a_i b_i
    //!
    double VectorDot(
        const dvec& a, 
        const dvec& b)
    {
        int N = a.size();
        return ddot_(&N, a.data(), &one, b.data(), &one);
    }
    
    //!
    //! Get the signs of the vector elements
    //!
    void VectorSgn(
        dvec& sgnVec,
        const dvec& vec)
    {
        auto it_sgn = sgnVec.begin();
        for(auto it_vec = vec.begin; it_vec < vec.end(); ++it_vec, ++it_sgn)
        {
            *it_sgn = (*it_vec > 0) - (*it_vec < 0);
        }
    }
    
    //!
    //! Get the L2 sum of vector elements
    //!
    double VectorL2(
        const dvec& a)
    {
        int N = a.size();
        return ddot_(&N, a.data(), &one, a.data(), &one);
    }
    
    //!
    //! Compute the L1 norm: output = sum_i |a_i|
    //!
    double VectorL1( 
        const dvec& a)
    {
        int N = a.size();
        return dasum_(&N, a.data(), &one);
    }
    
    //!
    //! Rescale the input factor by the scale factor
    //!
    void VectorScale(
        dvec& output, 
        const double scale,
        const dvec& input)
    {
        std::fill(output.begin(), output.end(), 0.0);
        int N = input.size();
        daxpy_(&N, &scale, input.data(), &one, output.data(), &one);
    }
}   //  End namespace utilities
