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

#ifndef _LOSS_FUNCTION_HPP_INCLUDED_
#define _LOSS_FUNCTION_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../utilities/general/dvec_def.hpp"

namespace ann
{
    //!
    //! Container for loss function weights
    //!
    struct LossFunctionWeights
    {
        bool usingResidualWeights;//!<  Flag set if using residual weights
        dvec residualWeights;   //!<    Weights for squared residuals of each sample
        double l1Alpha;         //!<    L1 constraint weight on alphas
        double l1Beta;          //!<    L1 constraint on betas
        double l2Alpha;         //!<    L2 constraint on alphas
        double l2Beta;          //!<    L2 constraint on betas
        LossFunctionWeights();
        void SetResidualWeights(const dvec& residualWeights);
    };
}   //  End namespace ann
#endif
