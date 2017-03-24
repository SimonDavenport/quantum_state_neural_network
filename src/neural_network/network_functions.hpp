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

#ifndef _NETWORK_FUNCTIONS_HPP_INCLUDED_
#define _NETWORK_FUNCTIONS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <cmath>

namespace ann
{
    double ShiftedExponential(const double& input);
    double ShiftedExponentialDeriv(const double& input);
    double Unit(const double& input);
    double UnitDeriv(const double& input);
    double Logistic(const double& input);
    double LogisticDeriv(const double& input);
}   //  End namespace ann
#endif
