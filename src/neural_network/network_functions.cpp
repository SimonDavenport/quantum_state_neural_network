////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains definitions of various functions that can be used
//!     in neural network constructions
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
#include "network_functions.hpp"

namespace ann
{
    //!
    //! Shifted exponential function
    //!
    double ShiftedExponential(const double& input)
    {
        return exp(input)-1.0;
    }
    
    //!
    //! Shifted exponential function derivative
    //!
    double ShiftedExponentialDeriv(const double& input)
    {
        return exp(input);
    }
    
    //!
    //! Unit function
    //!
    double Unit(const double& input)
    {
        return input;
    }
    
    //!
    //! Unit function derivative
    //!
    double UnitDeriv(const double& input)
    {
        return 1.0;
    }
    
    //!
    //! Logistic function
    //!
    double Logistic(const double& input)
    {
        return 1.0 / (1.0 + exp(-input));
    }
    
    //!
    //! Logistic function derivative
    //!
    double LogisticDeriv(const double& input)
    {
        double logistic = Logistic(input);
        return logistic*(1.0 - logistic);
    }
}   //  End namespace ann
