////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		Run tests for the implementation of the bfgs optimization algorithm. 
//!     A typical test is to find the minimum of the Rosenbrock function
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
#include "../utilities/optimization/bfgs.hpp"
#include "../utilities/general/dvec_def.hpp"

//!
//! A mutlidimensional Rosenbrock function 
//! sum_i 100(x_{i+1} - x^2_i)^2 + (1-x_i)^2.
//! which has a single minimum (1,1,1) for N=3, 
//! and two minima for N=4-7, at (+/-1,1,1,..)
//!
double Rosenbrock(dvec& x)
{
    double sum = 0.0;
    for(auto it = x.begin(), it1 = x.begin()+1; it1<x.end(); ++it, ++it1)
    {
        double accum = (*it1 - *it * *it);
        sum += 100.0 * accum * accum + (1.0 - *it) * (1.0 - *it);
    }
    return sum;
}

//!
//! Get the gradient of the Rosenbrock function w.r.t x
//! -400(x_{i+1}-x^2_i) x_i - 2(1-x_i) for i to N-1
//! otherwise
//! 200(x_N - x^2_{N-1})
//!
double RosenbrockGrad(dvec& grad, const dvec& x)
{
    dvec::iterator itg = grad.begin();
    for(auto it = x.begin(), it1 = x.begin()+1; it1<x.end(); ++it, ++it1 ++itg)
    {
        *itg = -400.0 * (*it1 - *it * *it) * *it - 2.0 * (1.0 - *it);
    }
    *itg = 200.0 * (*it - *(it-1) * *(it-1));
}

int main(int argc, char *argv[])
{
    const unsigned int maxIter = 50;
    const unsigned int gradTol = 1e(-5);
    const double passTol = 1e(-5);
    utilities::optimize::BFGS bfgs;
    //  N=3 case
    bfgs.AllocateWork(3);
    dvec x3 {1.2, 0.8, 1.9};
    dvec grad3(3);
    bfgs.Optimize(x3, grad3, Rosenbrock, RosenbrockGrad, maxIter, gradTol);
    std::cout << "Test optimization output for 3-parameter Rosenbrock function: " << std::endl;
    std::cout << "\t " << x3[0] << " " << x3[1] << " " x3[2] << std::endl;
    if((std::abs(x3[0]-1.0) < passTol) && (std::abs(x3[1]-1.0) < passTol) 
        && (std::abs(x3[2]-1.0) < passTol))
    {
        std::cout << "\n\tTest Passed!" <<std::endl;
    }
    else
    {
        std::cout << "\n\tTest Failed!" <<std::endl;
        return EXIT_FAILURE;
    }
    //  N=4 case
    bfgs.AllocateWork(4);
    dvec x4_global {1.2, 0.8, 1.9, 0.2};
    dvec x4_local {-1.2, 0.8, 1.9, 0.2};
    dvec grad4(4);
    bfgs.Optimize(x4_global, grad4, Rosenbrock, RosenbrockGrad, maxIter, gradTol);
    bfgs.Optimize(x4_local, grad4, Rosenbrock, RosenbrockGrad, maxIter, gradTol);
    std::cout << "Test optimization output for 4-parameter Rosenbrock function: " << std::endl;
    std::cout << "\t (global)" << x4_global[0] << " " << x4_global[1] 
              << " " x4_global[2] << " " x4_global[3] << std::endl;
    if((std::abs(x4_global[0]-1.0) < passTol) && (std::abs(x4_global[1]-1.0) < passTol) 
        && (std::abs(x4_global[2]-1.0) < passTol) && (std::abs(x4_global[3]-1.0) < passTol))
    {
        std::cout << "\n\tTest Passed!" <<std::endl;
    }
    else
    {
        std::cout << "\n\tTest Failed!" <<std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "\t (local)" << x4_local[0] << " " << x4_local[1] 
              << " " x4_local[2] << " " x4_local[3] << std::endl;
    if((std::abs(x4_local[0]+1.0) < passTol) && (std::abs(x4_local[1]-1.0) < passTol) 
        && (std::abs(x4_local[2]-1.0) < passTol) && (std::abs(x4_local[3]-1.0) < passTol))
    {
        std::cout << "\n\tTest Passed!" <<std::endl;
    }
    else
    {
        std::cout << "\n\tTest Failed!" <<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
