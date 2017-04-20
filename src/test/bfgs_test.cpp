////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		Run tests for the implementation of the bfgs and l-bfgs optimization 
//!     algorithms. A typical test is to find the minimum of the 
//!     multi-dimensional Rosenbrock function.
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
#include "../utilities/optimization/lbfgs.hpp"
#include "../utilities/general/dvec_def.hpp"
#include <iostream>

//!
//! A mutlidimensional Rosenbrock function 
//! sum_i 100(x_{i+1} - x^2_i)^2 + (1-x_i)^2.
//! which has a single minimum (1,1,1) for N=3, 
//! and two minima for N=4-7, at (+/-1,1,1,..)
//!
double Rosenbrock(const dvec& x)
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
//! For the first value:
//! -400(x_{2}-x^2_1) x_i - 2(1-x_1)
//! Between 2 and N-1:
//! -400(x_{i+1}-x^2_i) x_i - 2(1-x_i) + 200 (x_i - x^2_{i-1})
//! and the final value is:
//! 200(x_N - x^2_{N-1}) 
//!
void RosenbrockGrad(dvec& grad, const dvec& x)
{
    grad[0] = -400.0 * (x[1] - x[0] * x[0]) * x[0] - 2.0 * (1.0 - x[0]);
    dvec::iterator itg = grad.begin()+1;
    for(auto itm1 = x.begin(), it = x.begin()+1, it1 = x.begin()+2; it1<x.end(); ++itm1, ++it, ++it1, ++itg)
    {
        *itg = -400.0 * (*it1 - *it * *it) * *it - 2.0 * (1.0 - *it) + 200 * (*it - *itm1 * *itm1);
    }
    *itg = 200.0 * (*(x.end()-1) - *(x.end()-2) * *(x.end()-2));
}

int main(int argc, char *argv[])
{
    std::cout << "TEST BFGS IMPLEMENTATION" << std::endl;
    {
        const unsigned int maxIter = 100;
        double gradTol = 1e-7;
        double passTol = 1e-5;
        std::function<double(const dvec&)> minFunc = Rosenbrock;
        std::function<void(dvec&, const dvec&)> gradFunc = RosenbrockGrad;
        utilities::optimize::BFGS bfgs;
        //  N=3 case
        bfgs.AllocateWork(3);
        dvec x3 {1.2, 0.1, 5.5};
        dvec grad3(3);
        bfgs.Optimize(x3, grad3,  minFunc, gradFunc, maxIter, gradTol);
        std::cout << "Test optimization output for 3-parameter Rosenbrock function: " << std::endl;
        std::cout << "(global min) " << x3[0] << " " << x3[1] << " " << x3[2] << std::endl;
        if((std::abs(x3[0]-1.0) < passTol) && (std::abs(x3[1]-1.0) < passTol) 
            && (std::abs(x3[2]-1.0) < passTol))
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
        //  N=4 case
        bfgs.AllocateWork(4);
        dvec x4_global {1.4, 0.1, 3.5, 3.3};
        dvec x4_local {-1.2, 0.8, 1.0, 0.9};
        dvec grad4(4);
        bfgs.Optimize(x4_global, grad4, minFunc, gradFunc, maxIter, gradTol);
        gradTol = 1e-10;
        bfgs.Optimize(x4_local, grad4, minFunc, gradFunc, maxIter, gradTol);
        std::cout << "Test optimization output for 4-parameter Rosenbrock function: " << std::endl;
        std::cout << "(global min) " << x4_global[0] << " " << x4_global[1] 
                  << " " << x4_global[2] << " " << x4_global[3] << std::endl;
        if((std::abs(x4_global[0]-1.0) < passTol) && (std::abs(x4_global[1]-1.0) < passTol) 
            && (std::abs(x4_global[2]-1.0) < passTol) && (std::abs(x4_global[3]-1.0) < passTol))
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "(local min) " << x4_local[0] << " " << x4_local[1] 
                  << " " << x4_local[2] << " " << x4_local[3] << std::endl;
        if((std::abs(x4_local[0]+0.77565923) < passTol) && (std::abs(x4_local[1]-0.61309337) < passTol) 
            && (std::abs(x4_local[2]-0.38206285) < passTol) && (std::abs(x4_local[3]-0.14597202) < passTol))
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "TEST L-BFGS IMPLEMENTATION" << std::endl;
    {
        const unsigned int maxIter = 100;
        double gradTol = 1e-10;
        double passTol = 1e-5;
        std::function<double(const dvec&)> minFunc = Rosenbrock;
        std::function<void(dvec&, const dvec&)> gradFunc = RosenbrockGrad;
        utilities::optimize::LBFGS lbfgs;
        lbfgs.SetUpdateNumber(12);  //  Optimal case for this problem
        //  N=3 case
        lbfgs.AllocateWork(3);
        dvec x3 {1.4, 0.2, 1.7};
        dvec grad3(3);
        lbfgs.Optimize(x3, grad3,  minFunc, gradFunc, maxIter, gradTol);
        std::cout << "Test optimization output for 3-parameter Rosenbrock function: " << std::endl;
        std::cout << "(global min) " << x3[0] << " " << x3[1] << " " << x3[2] << std::endl;
        if((std::abs(x3[0]-1.0) < passTol) && (std::abs(x3[1]-1.0) < passTol) 
            && (std::abs(x3[2]-1.0) < passTol))
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
        //  N=4 case
        lbfgs.AllocateWork(4);
        dvec x4_global {1.2, 0.8, 1.9, 0.2};
        dvec x4_local {-1.2, 0.8, 1.0, 0.9};
        dvec grad4(4);
        lbfgs.Optimize(x4_global, grad4, minFunc, gradFunc, maxIter, gradTol);
        lbfgs.Optimize(x4_local, grad4, minFunc, gradFunc, maxIter, gradTol);
        std::cout << "Test optimization output for 4-parameter Rosenbrock function: " << std::endl;
        std::cout << "(global min) " << x4_global[0] << " " << x4_global[1] 
                  << " " << x4_global[2] << " " << x4_global[3] << std::endl;
        if((std::abs(x4_global[0]-1.0) < passTol) && (std::abs(x4_global[1]-1.0) < passTol) 
            && (std::abs(x4_global[2]-1.0) < passTol) && (std::abs(x4_global[3]-1.0) < passTol))
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "(local min) " << x4_local[0] << " " << x4_local[1] 
                  << " " << x4_local[2] << " " << x4_local[3] << std::endl;
        if((std::abs(x4_local[0]+0.77565923) < passTol) && (std::abs(x4_local[1]-0.61309337) < passTol) 
            && (std::abs(x4_local[2]-0.38206285) < passTol) && (std::abs(x4_local[3]-0.14597202) < passTol))
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
