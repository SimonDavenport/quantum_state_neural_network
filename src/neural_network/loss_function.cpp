////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains parameters defining the network loss function
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
#include "loss_function.hpp"

namespace ann
{
    //!
    //! Default constructor for loss function weights
    //!
    LossFunctionWeights::LossFunctionWeights()
        :
        usingResiduals(false),
        l1Alpha(0.0),
        l1Beta(0.0),
        l2Alpha(0.0),
        l2Beta(0.0)
    {}
    
    //!
    //! Set the residuals weights
    //!
    void LossFunctionWeights::SetResidualWeights(
        const dvec& residuals)
    {
        this->residuals = residuals;
        this->usingResiduals = true;
    }
}   //  End namespace ann
