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

#ifndef _DENSE_MATRIX_HPP_INCLUDED_
#define _DENSE_MATRIX_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../general/dvec_def.hpp"
#include "dense_vector.hpp"
#include "blas.hpp"

namespace utilities
{
    static int one = 1;
    
    template<typename T>
    struct matrix
    {
        unsigned int m_dLeading;    //!<    Leading dimension of matrix
        unsigned int m_dSecond;     //!<    Second dimension of matrix
        std::vector<T> m_data;      //!<    Container for matrix data
        
        //!
        //! Allocate space to store a n*m dense matrix
        //!
        matrix(const unsigned int dLeading, const unsigned int dSecond)
        {
            this->resize(dLeading, dSecond);
        }
        
        //!
        //! Resize the matrix
        //!
        void resize(const unsigned int dLeading, const unsigned int dSecond)
        {
            m_dLeading = dLeading;
            m_dSecond = dSecond;
            m_data.resize(m_dLeading*m_dSecond);
        }
        
        //!
        //! Get a pointer to the underlying container
        //!
        T* data()
        {
            return m_data.data();
        }
        
        //!
        //! Get the iterator at the start of the container
        //!        
        std::vector<T>::iterator begin()
        {
            return m_data.begin();
        }
        
        //!
        //! Get the iterator at the end of the container
        //!        
        std::vector<T>::iterator end()
        {
            return m_data.end();
        }
    };
    
    void SetToRandomMatrix(matrix<double>& mat, const double scale, const unsigned int seed);
    void SetToConstantMatrix(matrix<double>& mat, const double value);
    void SetToIdentityMatrix(matrix<double>& mat, const double value);
    void MatrixVectorMultiply(dvec& output, const double scale, matrix<double>& a, dvec& x);
    void SymmetricMatrixVectorMultiply(dvec& output, const double scale, matrix<double>& a, dvec& x);
    void MatrixMatrixMultiply(matrix<double>& output, matrix<double>& a, matrix<double>& b);
    void OuterProductIncrement(matrix<double>& a, const double scale, dvec& x, dvec& y);
    void SymmetricOuterProductIncrement(matrix<double>& a, const double scale, dvec& x, dvec& y);
    void MatrixHadamard(matrix<double>& output, matrix<double> a, matrix<double>& b);
    void MatrixIncrement(matrix<double>& a, double& scale, matrix<double>& b);
    void MatrixSgn(matrix<double>& sgnMat, matrix<double>& mat);
    void MatrixMask(matrix<double>& mat, std::vector<unsigned int>& zeros);
    double MatrixL2(dvec& mat);
    double MatrixL1(dvec& mat);
}   //  End namespace utilities
#endif
