////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains an implementation of a single-layer perceptron
//!     neural network
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

#ifndef _SINGLE_LAYER_PERCEPTRON_HPP_INCLUDED_
#define _SINGLE_LAYER_PERCEPTRON_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <iostream>
#include <functional>
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/linear_algebra/dense_vector.hpp"
#include "../utilities/optimization/bfgs.hpp"
#include "../utilities/general/dvec_def.hpp"
#if _DEBUG_
#include "../utilities/general/debug.hpp"
#endif

namespace ann
{
    //  Possible activation and output functions and their derivatives
    double ShiftedExponential(const double& input);
    double ShiftedExponentialDeriv(const double& input);
    double Unit(const double& input);
    double UnitDeriv(const double& input);
    
    //!
    //! Container for loss function weights
    //!
    struct LossFunctionWeights
    {
        bool usingResiduals;    //!<    Flag set if using weighted residuals
        dvec residuals;         //!<    Weights for squared residuals of each sample
        double l1Alpha;         //!<    L1 constraint weight on alphas
        double l1Beta;          //!<    L1 constraint on betas
        double l2Alpha;         //!<    L2 constraint on alphas
        double l2Beta;          //!<    L2 constraint on betas
        LossFunctionWeights();
        void SetResidualWeights(const dvec& residuals);
    };

    ////////////////////////////////////////////////////////////////////////////////
    //! \brief  Implementation of a single layer perceptron neural network
    //!
    //! Assuming N input vectors x_1,...,x_n of size P features and a single output
    //! The number of hidden nodes is H. The network output is given by computing
    //! Z = ActivationFunction(alpha^T X)
    //! Y = OutputFunction(beta^T Z)
    //!
    //! Where X are inputs is a P by N matrix of N input samples of P features. 
    //! alpha is of size H by P, Z is of size H by N, beta is of size H, 
    //! and Y is the output of size N. 
    ////////////////////////////////////////////////////////////////////////////////
    class SingleLayerPerceptron
    {
        private:
        utilities::matrix<double> m_alpha;  //!<    Set of H by P activation function weights
        std::vector<unsigned int> m_zeros;  //!<    List of locations of zero alpha weights
        dvec m_beta;                        //!<    Set of H output function weights
        unsigned int m_P;                   //!<    Number of features
        unsigned int m_H;                   //!<    Number of hidden nodes
        unsigned int m_N;                   //!<    Number of samples
        LossFunctionWeights m_lfWeights;    //!<    Instance of the loss function weights struct
        unsigned int m_workAllocated;       //!<    Flag work space allocated
        utilities::matrix<double> m_alphaGradient;
                                            //!<    Working space for alpha weights gradient
        dvec m_betaGradient;                //!<    Working space for beta weights gradient
        utilities::matrix<double> m_Z;      //!<    Working space for hidden parameters
        dvec m_output;                      //!<    Working space for network output
        dvec m_outputDeriv;                 //!<    Working space for network derivative output
        dvec m_residual;                    //!<    Working space for residual
        dvec m_sqResidual;                  //!<    Working space for squared residuals
        dvec m_delta;                       //!<    Working space for delta
        utilities::matrix<double> m_activationDeriv;
                                            //!<    Working space for activation function derivative
        utilities::matrix<double> m_S;      //!<    Working space for S matrix
        utilities::matrix<double> m_sgnAlpha;//!<    Working space for alpha weights signs
        dvec m_sgnBeta;                     //!<    Working space for beta weights signs
        std::function<double(const double& x)> m_ActivationImpl; 
                                            //!<    Activation function implementation
        std::function<double(const double& x)> m_ActivationDerivImpl; 
                                            //!<    Activation function derivative implementation
        std::function<double(const double& z)> m_OutputFunctionImpl;
                                            //!<    Output function implementation
        std::function<double(const double& z)> m_OutputFunctionDerivImpl;
                                            //!<    Output function derivative implementation
        void ActivationFunction(utilities::matrix<double>& Z, 
                                const utilities::matrix<double>& X);
        void ActivationFunctionDeriv(utilities::matrix<double>& Z, 
                                     const utilities::matrix<double>& X);
        void OutputFunction(dvec& Y, const utilities::matrix<double>& Z);
        void OutputFunctionDeriv(dvec& Y, const utilities::matrix<double>& Z);
        public:
        SingleLayerPerceptron();
        SingleLayerPerceptron(const unsigned int P, const unsigned int H);
        ~SingleLayerPerceptron();
        void AllocateWork(unsigned int N);
        bool CheckDimensions(const dvec& Y, const utilities::matrix<double>& X);
        void RandomizeWeights(const double scale, const unsigned int seed);
        void SetActivationFunction(
            std::function<double(const double& x)> activationImpl,
            std::function<double(const double& x)> activationDerivImpl);
        void SetLossFunctionWeights(const LossFunctionWeights& lfWeights);
        void SetWeights(const utilities::matrix<double>& alpha, const dvec& beta);
        void GetWeights(utilities::matrix<double>& alpha, dvec& beta) const;
        void SetZeros(const std::vector<unsigned int>& zeros);
        void GetZeros(std::vector<unsigned int>& zeros) const;
        unsigned int nnzWeights() const;
        unsigned int nnzAlpha() const;
        void UpdateNzWeights(const dvec& nzWeights);
        void ExtractNzWeights(dvec& nzWeights) const;
        void ExtractNzGradients(dvec& nzGradients) const;
        void Evaluate(dvec& Y, const utilities::matrix<double>& X);
        double EvaluateSquaredLoss(const dvec& Y, const utilities::matrix<double>& X);
        void EvaluateSquaredLossGradient(const dvec& Y, const utilities::matrix<double>& X);
        
    };
    
    //  Helper functions for training the network
    double EvaluateSquaredLoss(const dvec& nzWeights, SingleLayerPerceptron& slp, 
                               const dvec& Y, const utilities::matrix<double>& X);
    void EvaluateNzSquaredLossGradient(dvec& nzGradients, const dvec& nzWeights,
                                       SingleLayerPerceptron& slp, 
                                       const dvec& Y, const utilities::matrix<double>& X);
    void Train(SingleLayerPerceptron& slp, const dvec& Y, 
        const utilities::matrix<double>& X);
    void Train(SingleLayerPerceptron& slp, const dvec& Y, 
        const utilities::matrix<double>& X, const unsigned int maxIter, 
        const double gradTol);
}   //  End namespace ann
#endif
