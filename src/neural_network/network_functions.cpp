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
    
    //!
    //! Tanh function
    //!
    double Tanh(const double& input)
    {
        return tanh(input);
    }
    
    //!
    //! Tanh function derivative
    //!
    double TanhDeriv(const double& input)
    {
        double tanhVal = tanh(input);
        return 1.0 - tanhVal * tanhVal;
    }

    //!
    //! Rectified linear unit funtion
    //!
    double RectifiedLinearUnit(const double& input)
    {
        return std::max(0.0, input);
    }
    
    //!
    //! Rectified linear unit derivative funtion
    //!
    double RectifiedLinearUnitDeriv(const double& input)
    {
        if(input < 0.0)
        {
            return 0.0;
        }
        else
        {
            return 1.0;
        }
    }

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
    //! Select activation function from a list of those implemented
    //!
    void SelectActivation(
        std::function<double(const double& x)>& ActivationImpl,
        std::function<double(const double& x)>& ActivationDerivImpl,
        const std::string activationFuncName)
    {
        std::map<std::string, std::function<double(const double& x)>> implFunctions;
        std::map<std::string, std::function<double(const double& x)>> implDerivFunctions;
        //  Define map of possible functions
        implFunctions["identity"] = Unit;
        implDerivFunctions["identity"] = UnitDeriv;
        implFunctions["logistic"] = Logistic;
        implDerivFunctions["logistic"] = LogisticDeriv;
        implFunctions["tanh"] = Tanh;
        implDerivFunctions["tanh"] = TanhDeriv;
        implFunctions["relu"] = RectifiedLinearUnit;
        implDerivFunctions["relu"] = RectifiedLinearUnitDeriv;
        implFunctions["shifted-exp"] = ShiftedExponential;
        implDerivFunctions["shifted-exp"] = ShiftedExponentialDeriv;
        //  Set implementation from map
        ActivationImpl = implFunctions[activationFuncName];
        ActivationDerivImpl = implDerivFunctions[activationFuncName];

        //std::cerr << "ERROR no known activation function: " 
        //          << activationFuncName << ". Defaulting to logistic" << std::endl;
        //ActivationImpl = Logistic;
        //ActivationDerivImpl = LogisticDeriv;
        //break;
    }
}   //  End namespace ann
