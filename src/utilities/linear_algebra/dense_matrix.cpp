////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains an implementation of a generic dense matrix 
//!     container implementation and some related linear algebra
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

#include "dense_matrix.hpp"

namespace utilities
{
    //!
    //! Set to a random matrix using a given seed
    //!
    void SetToRandomMatrix(
        matrix<double>& mat, 
        const double scale, 
        const unsigned int seed)
    {
        SetToRandomVector(mat.m_data, scale, seed);
    }
    
    //!
    //! Set all matrix elements to a constant
    //!
    void SetToConstantMatrix(
        matrix<double>& mat,
        const double value)
    {
        std::fill(mat.begin(), mat.end(), value);
    }
    
    //!
    //! Set all matrix elements to an identity matrix
    //!
    void SetToIdentityMatrix(
        matrix<double>& mat,
        const double value)
    {
        for(auto it = mat.begin(); it < mat.end(); it += mat.m_dLeading)
        {
            *it = 1.0;
        }
    }
    
    //!
    //! Standard matrix-vector multiplication algorithm output = ALPHA*a*x
    //!
    void MatrixVectorMultiply(
        dvec& output, 
        const double scale,
        matrix<double>& a, 
        dvec& x)
    {
        int M = output.size();
        int N = x.size(); 
        double BETA = 0.0;
        dgemv_('N', &M, &N, &scale, a.data(), &M, x.data(), &one, &BETA, x.data(), &one);
    }
    
    //!
    //! Symmetric matrix-vector multiplication algorithm output = ALPHA*a*x
    //!
    void SymmetricMatrixVectorMultiply(
        dvec& output, 
        const double scale,
        matrix<double>& a, 
        dvec& x)
    {
        int N = x.size(); 
        double BETA = 0.0;
        dsymv_('U', &N, &scale, a.data(), &N, x.data(), &one, &BETA, x.data(), &one);
    }
    
    //!
    //! Standard matrix-matrix multiplication algorithm C := ALPHA*A*B
    //! A and B can be optionally transposed
    //!
    void MatrixMatrixMultiply(
        matrix<double>& output, 
        matrix<double>& a, 
        matrix<double>& b, 
        std::string trOpt)
    {
        char TRANSA = trOpt[0];
        char TRANSB = trOpt[1];
        int M = a.m_dLeading;
        int N = b.m_dSecond;
        int K = b.m_dLeading;
        double ALPHA = 1.0;
        double BETA = 0.0;
        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, a.data(), &M, b.data(), &K, &BETA, &M);
    }

    //!
    //! Increment by the outer product of two vectors 
    //! a += ALPHA*x*y^T
    //!
    void OuterProductIncrement(
        matrix<double>& a, 
        const double scale,
        dvec& x, 
        dvec& y)
    {
        int M = x.size();
        int N = y.size();
        dger_(&M, &N, &scale, &x, &one, &y, &one, a.data(), &M);
    }

    //!
    //! Increment by the outer product of two vectors 
    //! a += ALPHA*x*x^T 
    //!
    void SymmetricOuterProductIncrement(
        matrix<double>& a, 
        const double scale,
        dvec& x)
    {
        char UPLO = 'U';
        int N = x.size();
        dsyr_(&UPLO, &N, &scale, &x, &one, a.data(), &N);
    }
    
    //!
    //! Increment by the outer product of two vectors
    //! a += ALPHA*x*y^T + ALPHA*y*x^T 
    //!
    void SymmetricOuterProductIncrement(
        matrix<double>& a, 
        const double scale,
        dvec& x, 
        dvec& y)
    {
        char UPLO = 'U';
        int N = x.size();
        dsyr2_(&UPLO, &N, &scale, &x, &one, &y, &one, a.data(), &N);
    }

    //!
    //! Multiply two matrices element-wise and write to output
    //! output_ij = a_ij * b_ij
    //!
    void MatrixHadamard(
        matrix<double>& output, 
        const double scale, 
        matrix<double>& a, 
        matrix<double>& b)
    {
        VectorHadamard(output.m_data, scale, a.m_data, b.m_data);
    }

    //!
    //! Compute a := a + scale*b
    //!
    void MatrixIncrement(
        matrix<double>& a,  
        double& scale, 
        matrix<double>& b)
    {
        VectorIncrement(a.m_data, scale, b.m_data);
    }
    
    //!
    //! Get the signs of the matrix elements
    //!
    void MatrixSgn(
        matrix<double>& sgnMat, 
        matrix<double>& mat)
    {
        VectorSgn(sgnMat.m_data, mat.m_data);
    }
    
    //!
    //! Set a list of elements to be zero
    //!
    void MatrixMask(
        matrix<double>& mat, 
        std::vector<unsigned int>& zeros)
    {
        auto it_mat = mat.begin()
        unsigned int runSize = 0;
        for(auto it_zeros = zeros.begin(); it_zeros < zeros.end(); ++it_zeros)
        {
            runSize = *it_zeros - runSize;
            it_mat += d_index;
            *it_mat = 0;
        }
    }
 
    //!
    //! Get the L2 sum of matrix elements
    //!
    double MatrixL2(
        matrix<double>& mat)
    {
        return VectorL2(mat.m_data);
    }
    
    //!
    //! Get the L1 sum of matrix elements
    //!
    double MatrixL1(
        matrix<double>& mat)
    {
        return VectorL1(mat.m_data);
    }
};
