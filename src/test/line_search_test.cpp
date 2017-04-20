////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		Run tests for the implementation line search algorithms and their
//!     constituent parts. 
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
#include "../utilities/optimization/line_search.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "TEST CUBIC POLYNOMIAL MINIMIZER" << std::endl;
    {
        //  Find the minimizer for a cubic polynomial that goes through 
        //  the points (a, fa), (b, fb) and (c,fc) with derivative fpa
        //  at a. 
        double a = 1;
        double fa = 1;
        double fpa = 1;
        double b = 2;
        double fb = 2;
        double c = -1;
        double fc = 1;
        std::cout << "(a, fa) = " << a << " , " << fa << std::endl;
        std::cout << "fpa = " << fpa << std::endl;
        std::cout << "(b, fb) = " << b << " , " << fb << std::endl;
        std::cout << "(c, fc) = " << c << " , " << fc << std::endl;
        double xmin = utilities::optimize::CubicMin(a, fa, fpa, b, fb, c, fc);
        std::cout << "xmin = " << xmin << std::endl;
        if(abs(-0.11963298118022458 - xmin) < 1e-15)
        {
            std::cout << "Test Passed!" <<std::endl;
        }
        else
        {
            std::cout << "Test Failed!" <<std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "TEST QUADRATIC POLYNOMIAL MINIMIZER" << std::endl;
    {
        //  Find the minimizer for a quadratic polynomial that goes through
        //  the points (a,fa), (b,fb) with derivative fpa at a.
        double a = 1;
        double fa = 1;
        double fpa = 1;
        double b = -2;
        double fb = 2;
        std::cout << "(a, fa) = " << a << " , " << fa << std::endl;
        std::cout << "fpa = " << fpa << std::endl;
        std::cout << "(b, fb) = " << b << " , " << fb << std::endl;
        double xmin = utilities::optimize::QuadMin(a, fa, fpa, b, fb);
        std::cout << "xmin = " << xmin << std::endl;
        if(abs(-0.125 - xmin) < 1e-15)
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
