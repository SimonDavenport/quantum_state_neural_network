////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains some wrappers for generic network training
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

#ifndef _TRAIN_HPP_INCLUDED_
#define _TRAIN_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/linear_algebra/dense_vector.hpp"
#include "../utilities/general/dvec_def.hpp"

namespace ann
{
    //!
    //! Evaluate squared loss given a vector of non-zero
    //! network weights
    //!
    template<class N>
    double EvaluateSquaredLoss(
        const dvec& nzWeights,          //!<    Non-zero network weights
        N& network,                     //!<    Network to be trained
        const dvec& Y,                  //!<    Vector of N training outputs
        const utilities::matrix<double>& X)//!<    P by N array of training inputs    
    {
        network.SetNzWeights(nzWeights);
        return network.EvaluateSquaredLoss(Y, X);
    }

    //!
    //! Evaluate gradient of square loss function given
    //! a vector of non zero network weights, and to extract non-zero
    //! gradients of those weights
    //! 
    template<class N>
    void EvaluateSquaredLossGradient(
        dvec& nzGradients,          //!<    Non-zero network gradients
        const dvec& nzWeights,      //!<    Non-zero network weights
        N& network,                 //!<    Network to be optimized
        const dvec& Y,              //!<    Vector of N training outputs
        const utilities::matrix<double>& X)//!<    P by N array of inputs
    {
        network.SetNzWeights(nzWeights);
        network.EvaluateSquaredLossGradient(Y, X);
        network.GetNzGradients(nzGradients);
    }
    
    //!
    //! Train function using default optimization parameters
    //!
    template<class O, class N>
    void Train(
        N& network,                             //!<    Network to be trained
        O& optimizer,                           //!<    Optimization method
        const dvec& Y,                          //!<    Vector of N training outputs
        const utilities::matrix<double>& X)     //!<    P by N training inputs
    {
        Train(network, optimizer, Y, X, 50, 1e-5);
    }

    //!
    //! Train the network N using optimization method O
    //!
    template<class O, class N>
    void Train(
        N& network,                     //!<    Network to be trained
        O& optimizer,                   //!<    Optimization method
        const dvec& Y,                  //!<    Vector of N training outputs
        const utilities::matrix<double>& X,//!<    P by N array of training inputs
        const unsigned int maxIter,     //!<    Maximum number of allowed iteractions
                                        //!     of optimization
        const double gradTol)           //!<    Gradient tol for terminating bfgs
    {
        network.AllocateWork(Y.size());
        if(network.CheckDimensions(Y, X))
        {
            return;
        }
        dvec x(network.nnzWeights());
        dvec grad(network.nnzWeights());
        network.GetNzWeights(x);
        network.GetNzGradients(grad);
        optimizer.AllocateWork(network.nnzWeights());
        std::function<double(const dvec&)> minFunc = std::bind(EvaluateSquaredLoss<N>, std::placeholders::_1, network, Y, X);
        std::function<void(dvec&, const dvec&)> gradFunc = std::bind(EvaluateSquaredLossGradient<N>, std::placeholders::_1, std::placeholders::_2, network, Y, X);
        optimizer.Optimize(x, grad, minFunc, gradFunc, maxIter, gradTol);
        network.SetNzWeights(x);
    }
}   //  End namespace ann
#endif
