////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains function declarations implementing BLAS subroutines
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

#ifndef _BLAS_HPP_INCLUDED_
#define _BLAS_HPP_INCLUDED_

///////		Import selected subroutines from the BLAS library
extern "C"
{
//!
//! Copy a vector Y = X
//!
void ccopy_(const int* N, const double* X, const int* INCX, double* Y, 
            const int* INCY);
//!
//! Compute the dot product X.Y
//!
double ddot_(const int* N, const double* X, const int* INCX, const double* Y, 
             const int* INCY);
//!
//! Compute sum of abs values of X
//!
double dasum_(const int* N, const double* X, const int* INCX);
//!
//! Compute Y = Y + A*X for scalar A
//!
void daxpy_(const int* N, const double* A, const double* X, const int* INCX, 
            double* Y, const int* INCY);
//!
//! Increment by an outer product A := ALPHA*X*Y^T + A 
//! where X and Y are vectors
//!
void dger_(const int* M, const int* N, const double* ALPHA, 
           const double* X, const int* INCX, const double* Y,
           const int* INCY, double* A, const int* LDA);
//!
//! Increment by a symmetric outer product A:= ALPHA*X*X^T +A
//!
void dsyr_(const char* UPLO, const int* N, const double* ALPHA, 
           const double* X, const int* INCX, double* A, const int* LDA);
//!
//! Increment by a symmetric outer product A := ALPHA*X*Y^T + ALPHA*Y*X^T + A
//!
void dsyr2_(const char* UPLO, const int* N, const double* ALPHA, 
            const double* X, const int* INCX, const double* Y,
            const int* INCY, double* A, const int* LDA);
//!
//! Matrix-vector multiplication Y = ALPHA*A*X + BETA*Y for
//! scalars ALPHA and BETA, vectors X and Y and symmetric banded matrices A
//!
void dsbmv_(const char *UPLO, const int *N, const int *K, const double *ALPHA, 
            const double *A, const int *LDA, const double *X, const int *INCX,
            const double *BETA, double *Y, const int *INCY);
//!
//! Matrix-vector multiplication Y = ALPHA*A*X + BETA*Y for
//! scalars ALPHA and BETA, vectors X and Y and symmetric matrices A
//!
void dsymv_(const char *UPLO, const int *N, const double *ALPHA, 
            const double *A, const int *LDA, const double *X, const int *INCX,
            const double *BETA, double *Y, const int *INCY);
//!
//! Matrix-vector multiplication Y = ALPHA*A*X + BETA*Y for 
//! scalars ALPHA and BETA, vectors X and Y and general matrices
//!
void dgemv_(const char *TRANS, const int* M, const int* N, const double* ALPHA,
            const double* A, const int* LDA, const double* X, const int* INCX,
            const double* BETA, const double* Y, const int* INCY);
//!
//! General matrix multiplication C = ALPHA*A*B + BETA*C
//! Potentially A, B or both can be transposed
//!
void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, 
            const int* K, const double* ALPHA, const double* A, const int* LDA,
            const double* B, const int* LDB, double* BETA, double* C, const int* LDC);
}
#endif
