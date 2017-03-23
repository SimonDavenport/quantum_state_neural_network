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
        std::fill(mat.m_data.begin(), mat.m_data.end(), value);
    }
    
    //!
    //! Set to an identity matrix
    //!
    void SetToIdentityMatrix(
        matrix<double>& mat)
    {
        SetToConstantMatrix(mat, 0.0);
        for(auto it = mat.m_data.begin(); it < mat.m_data.end(); it += mat.m_dLeading+1)
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
        const matrix<double>& a, 
        const dvec& x,
        char TRANS)
    {
        static const int one = 1;
        int M = output.size();
        int N = x.size();
        double BETA = 0.0;
        dgemv_(&TRANS, &M, &N, &scale, a.data(), &M, x.data(), &one, &BETA, output.data(), &one);
    }
    
    //!
    //! Symmetric matrix-vector multiplication algorithm output = ALPHA*a*x
    //!
    void SymmetricMatrixVectorMultiply(
        dvec& output, 
        const double scale,
        const matrix<double>& a, 
        const dvec& x)
    {
        static const int one = 1;
        int N = x.size(); 
        double BETA = 0.0;
        dsymv_(&UPLO, &N, &scale, a.data(), &N, x.data(), &one, &BETA, output.data(), &one);
    }
    
    //!
    //! Standard matrix-matrix multiplication algorithm C := ALPHA*A*B
    //! A and B can be optionally transposed. 
    //!
    void MatrixMatrixMultiply(
        matrix<double>& c, 
        const matrix<double>& a, 
        const matrix<double>& b, 
        std::string trOpt)
    {
        char TRANSA = trOpt[0];
        char TRANSB = trOpt[1];
        int M, N, K, LDA, LDB;
        if('N' == TRANSA)
        {
            M = a.m_dLeading;
            K = a.m_dSecond;
            LDA = M;
        }
        else
        {
            M = a.m_dSecond;
            K = a.m_dLeading;
            LDA = K;
        }
        if('N' == TRANSB)
        {
            N = b.m_dSecond;
            LDB = K;
        }
        else
        {
            N = b.m_dLeading;
            LDB = N;
        }
        
        double ALPHA = 1.0;
        double BETA = 0.0;
        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, a.data(), &LDA, 
               b.data(), &LDB, &BETA, c.data(), &M);
    }

    //!
    //! Increment by the outer product of two vectors 
    //! a += ALPHA*x*y^T
    //!
    void OuterProductIncrement(
        matrix<double>& a, 
        const double scale,
        const dvec& x, 
        const dvec& y)
    {
        static const int one = 1;
        int M = x.size();
        int N = y.size();
        dger_(&M, &N, &scale, x.data(), &one, y.data(), &one, a.data(), &M);
    }

    //!
    //! Increment by the outer product of two vectors 
    //! a += ALPHA*x*x^T 
    //!
    void SymmetricOuterProductIncrement(
        matrix<double>& a, 
        const double scale,
        const dvec& x)
    {
        static const int one = 1;
        int N = x.size();
        dsyr_(&UPLO, &N, &scale, x.data(), &one, a.data(), &N);
    }
    
    //!
    //! Increment by the outer product of two vectors
    //! a += ALPHA*x*y^T + ALPHA*y*x^T 
    //!
    void SymmetricOuterProductIncrement(
        matrix<double>& a, 
        const double scale,
        const dvec& x, 
        const dvec& y)
    {
        static const int one = 1;
        int N = x.size();
        dsyr2_(&UPLO, &N, &scale, x.data(), &one, y.data(), &one, a.data(), &N);
    }

    //!
    //! Multiply two matrices element-wise and write to output
    //! c_ij = scale * a_ij * b_ij + BETA*c_ij
    //!
    void MatrixHadamard(
        matrix<double>& c, 
        const double scale,
        const matrix<double>& a, 
        const matrix<double>& b)
    {
        VectorHadamard(c.m_data, scale, a.m_data, b.m_data);
    }

    //!
    //! Compute a := a + scale*b
    //!
    void MatrixIncrement(
        matrix<double>& a,  
        const double& scale, 
        const matrix<double>& b)
    {
        VectorIncrement(a.m_data, scale, b.m_data);
    }
    
    //!
    //! Get the signs of the matrix elements
    //!
    void MatrixSgn(
        matrix<double>& sgnMat, 
        const matrix<double>& mat)
    {
        VectorSgn(sgnMat.m_data, mat.m_data);
    }
    
    //!
    //! Set a list of elements to be zero
    //!
    void MatrixMask(
        matrix<double>& mat, 
        const std::vector<unsigned int>& zeros)
    {
        auto it_mat = mat.m_data.begin();
        int nnzConsecutive = 0;
        for(auto it_zeros = zeros.begin(); it_zeros < zeros.end(); ++it_zeros)
        {
            nnzConsecutive = *it_zeros - nnzConsecutive;
            it_mat += nnzConsecutive;
            *it_mat = 0;
        }
    }
 
    //!
    //! Get the L2 sum of matrix elements
    //!
    double MatrixL2(
        const matrix<double>& mat)
    {
        return VectorL2(mat.m_data);
    }
    
    //!
    //! Get the L1 sum of matrix elements
    //!
    double MatrixL1(
        const matrix<double>& mat)
    {
        return VectorL1(mat.m_data);
    }
    
    //!
    //! Slice a given matrix to return a sub matrix. Using fortran-style indexing.
    //!
    void ToSubMatrix(
        matrix<double>& output, 
        const matrix<double>& input, 
        const unsigned int leadingOffset, 
        const unsigned int secondOffset)
    {    
        for(unsigned int inCol = leadingOffset, outCol = 0; inCol<output.m_dLeading; ++inCol, ++outCol)
        {
            CopyVector(&output.m_data[outCol*output.m_dSecond], 
                       &input.m_data[inCol*input.m_dSecond+secondOffset], output.m_dSecond);
        }
    }
};
