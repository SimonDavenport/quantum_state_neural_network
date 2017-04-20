////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		Run tests for the implementation of the single layer perceptron 
//!     neutral network.
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

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/linear_algebra/dense_vector.hpp"

#include "../utilities/general/debug.hpp"

int main(int argc, char *argv[])
{   
    const unsigned int N = 3;
    const unsigned int P = 3;
    const unsigned int H = 2;
    utilities::matrix<double> a(P, N);
    utilities::matrix<double> b(N, H);
    utilities::matrix<double> c(P, H);
    std::vector<double> d(H);
    std::vector<double> e(P);
    
    utilities::SetToRandomMatrix(a, 1.0, 0);
    utilities::SetToRandomMatrix(b, 2.0, 1);
    std::fill(d.begin(), d.end(), -0.5);
    
    utilities::MatrixMatrixMultiply(c, a, b, "NN");
    
    std::cout << "Compute C = A*B" << std::endl;
    
    PRINTVEC("A", a.m_data);
    PRINTVEC("B", b.m_data);
    PRINTVEC("C", c.m_data);
    
    //utilities::MatrixMatrixMultiply(c, a, b, "NT");
    //utilities::MatrixVectorMultiply(e, 1.0, c, d, 'N');
    
    //PRINTVEC("E", e);
    
    //utilities::MatrixVectorMultiply(e, 1.0, c, d, 'T');
    
    return EXIT_SUCCESS;
}
