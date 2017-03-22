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

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "single_layer_perceptron.hpp"

namespace neural_network
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
    //! Implementation of activation function, mapping N inputs X of size P to 
    //! H hidden node parameters Z via weights alpha.
    //!
    void SingleLayerPerceptron::ActivationFunction(
        utilities::matrix<double>& Z,    //!<    H by N hidden node parameters
        utilities::matrix<double>& X)    //!<    P by N inputs
    {
        utilities::MatrixMatrixMultiply(Z, m_alpha, X);
        std::for_each(Z.begin(), Z.end(), m_ActivationImpl);
    }
    
    //!
    //! Implementation of activation function derivative
    //!
    void SingleLayerPerceptron::ActivationFunctionDeriv(
        utilities::matrix<double>& Z,    //!<    H by N hidden node parameters
        utilities::matrix<double>& X)    //!<    P by N inputs
    {
        utilities::MatrixMatrixMultiply(Z, m_alpha, X);
        std::for_each(Z.begin(), Z.end(), m_ActivationDerivImpl);
    }
    
    //!
    //! Implementation of the output function, mapping hidden node parameters Z
    //! to output parameters Y via weights beta
    //!
    void SingleLayerPerceptron::OutputFunction(
        dvec& Y,                         //!<    Vector of N outputs
        utilities::matrix<double>& Z)    //!<    H by N hidden node parameters
    {
        utilities::VectorMatrixMultiply(Y, Z, m_beta);
        std::for_each(Y.begin(), Y.end(), m_OutputFunctionImpl);
    }
    
    //!
    //! Implementation of the output function derivative
    //!
    void SingleLayerPerceptron::OutputFunctionDeriv(
        dvec& Y,                         //!<    Vector of N outputs
        utilities::matrix<double>& Z)    //!<    H by N hidden node parameters
    {
        utilities::VectorMatrixMultiply(Y, Z, m_beta);
        std::for_each(Y.begin(), Y.end(), m_OutputFunctionDerivImpl);
    }        
    
    //!
    //! Check the dimensions of the input containers
    //!
    bool SingleLayerPerceptron::CheckDimensions(
        dvec& Y,                         //!<    Vector of N outputs
        utilities::matrix<double>& X)    //!<    P by N array of inputs
    {
        std::pair<unsigned int, unsigned int> xDim = X.size();
        if(xDim.second != Y.size())
        {
            std::cerr << "ERROR: X and Y dimension mismatch "<<std::endl;
            return true;
        }
        if(xDim.first != m_P)
        {
            std::cerr << "ERROR: X feature dimension mismatch "<<std::endl;
            return true;
        }
        return false;
    }
    
    //!
    //! Default constructor
    //!
    SingleLayerPerceptron::SingleLayerPerceptron()
        :
        m_H = 0,
        m_P = 0,
        m_N = 0,
        m_workAllocated = false,
        m_ActivationImpl = ShiftedExponential,
        m_ActivationDerivImpl = ShiftedExponentialDeriv,
        m_OutputFunctionImpl = Unit,
        m_OutputFunctionDerivImpl = UnitDeriv
    {}
    
    //! 
    //! Destructor
    //!
    SingleLayerPerceptron::~SingleLayerPerceptron()
    {}
    
    //!
    //! Function to initialize the network with a H hidden nodes and P features.
    //! Random weights are assigned.
    //!
    void SingleLayerPerceptron::RandomizeWeights(
        const unsigned int H,       //!<    Number of hidden nodes in layer
        const unsigned int P,       //!<    Number of features
        const double scale,         //!<    Scale of the initial random weights
        const unsigned int seed)    //!<    Random seed
    {
        m_H = H;
        m_P = P;
        m_alpha.resize(m_H, m_P);
        m_beta.resize(m_H);
        utilities::SetToRandomVector(m_alpha, seed);
        utilities::SetToRandomMatrix(m_beta, seed);
    }
    
    //!
    //! Set weights (note that the mask is imposed on the set weights)
    //!
    void SingleLayerPerceptron::SetWeights(
        const dvec& alpha,                      //!<    Updated alpha weights
        const utilities::matrix<double>& beta)  //!<    Updated beta weights
    {
        m_alpha = alpha;
        m_beta = beta;
        m_H = alpha.m_dLeading;
        m_P = alpha.m_dSecond;
        utilities::MatrixMask(m_alpha, m_zeros);
    }
  
    //!
    //! Get weights
    //!
    void SingleLayerPerceptron::GetWeights(
        dvec& alpha,                        //!<    Alpha weights container
        utilities::matrix<double>& beta)    //!<    Beta weights container
        const
    {
        alpha = m_alpha;
        beta = m_beta;
    }
    
    //!
    //! Set zero alpha weights locations
    //!
    void SingleLayerPerceptron::SetZeros(
        const std::vector<unsigned int>& zeros)  //!<   Locations of zeros
    {
        m_zeros = zeros;
        utilities::MatrixMask(m_alpha, m_zeros);
    }
    
    //!
    //! Get zero alpha weights locations
    //!
    void SingleLayerPerceptron::GetZeros(
        std::vector<unsigned int>& zeros)       //!<    Locations of zeros
        const
    {
        mask = m_zeros;
    }
    
    //!
    //! Get total number of non-zero alpha and beta weights
    //!
    unsigned int SingleLayerPerceptron::nnzWeights() const
    {
        return this->nnzAlpha() + m_H;
    }
    
    //!
    //! Get number of non-zero alpha weights
    //!
    unsigned int SingleLayerPerceptron::nnzAlpha() const
    {
        return m_H*m_P - m_zeros.size();
    }
    
    //!
    //! Update non-zero weights from a vector container
    //!
    void SingleLayerPerceptron::UpdateNzWeights(
        const dvec& nzWeights)          //! Vector container for non-zero 
                                        //! alpha and beta weights
    {
        utilities::ToSubVector(m_alpha.m_data, weights, 0, m_zeros);
        utilities::ToSubVector(m_beta, weights, this->nnzAlpha());
    }
    
    //!
    //! Get non-zero weights and place in a vector container
    //!
    void SingleLayerPerceptron::ExtractNzWeights(
        dvec& nzWeights)                //! Vector container for non-zero 
                                        //! alpha and beta weights
        const
    {
        utilities::FromSubVector(m_alpha.m_data, weights, 0, m_zeros);
        utilities::FromSubVector(m_beta, weights, this->nnzAlpha());
    }
    
    //!
    //! Get the gradients to a single vector container
    //!
    void SingleLayerPerceptron::ExtractNzGradients(
        dvec& gradients)                //!<  Vector container for non-zero 
                                        //!<  alpha and beta gradients
        const
    {
        std::vector<unsigned int> empty;
        utilities::FromSubVector(m_alphaGradient.m_data, gradients, 0, m_zeros);
        utilities::FromSubVector(m_betaGradient, gradients, this->nnzAlpha(), empty);
    }
    
    //!
    //! Allocate working space
    //!
    void SingleLayerPerceptron::AllocateWork(
        unsigned int N)     //!<    Working space to allocate for
    {
        m_N = N;
        Z.resize(m_H, m_N);
        outputDeriv.resize(m_N);
        output.resize(m_N);
        residual.resize(m_N);
        delta.resize(m_N);
        activationDeriv.resize(m_H, m_N);
        S.resize(m_H, m_N);
        alphaGradient.resize(m_H, m_P);
        betaGradient.resize(m_H);
        sgnAlpha.resize(m_H, m_P);
        sgnBeta.resize(m_H);
        m_workAllocated = true;
    }
    
    //!
    //! Evalute the current output of the neural network for given inputs X,
    //! where X contains N inputs of size P features in a P by N matrix, and write
    //! it to the vector Y.
    //!
    void SingleLayerPerceptron::Evaluate(
        dvec& Y,                                //!<    Vector of N outputs
        const utilities::matrix<double>& X)     //!<    P by N array of inputs
        const
    {
        if(!m_workAllocated)
        {
            unsigned int N = Y.size();
            m_Z.resize(m_H, N);
        }
        this->ActivationFunction(m_Z, X);
        this->OutputFunction(Y, m_Z);
    }
    
    //!
    //! Overload for evaulation of squared loss given a vector of non-zero
    //! network weights
    //!
    double SingleLayerPerceptron::EvaluateSquaredLoss(
        const dvec& nzWeights,                  //!<    Non-zero network weights
        const dvec& Y,                          //!<    Vector of N training outputs
        const utilities::matrix<double>& X,     //!<    P by N array of training inputs    
        const LossFunctionWeights& lfWeights)   //!<    Constraint weights in the loss function
    {
        this->UpdateNzWeights(nzWeights);
        return this->EvaluateSquaredLoss(Y, X, lfWeights);
    }
    
    //!
    //! Evaluate a squared loss function between the current output 
    //! and a training data set.
    //! L = sum_i residualWeights_i R_i^2 + l1AlphaWeight*sum|alpha| 
    //! + l1BetaWeight*sum|beta| + l2AlphaWeight*sum alpha^2 
    //! + l2BetaWeight*sum beta^2
    //!
    double SingleLayerPerceptron::EvaluateSquaredLoss(
        const dvec& Y,                          //!<    Vector of N training outputs
        const utilities::matrix<double>& X,     //!<    P by N array of training inputs    
        const LossFunctionWeights& lfWeights)   //!<    Constraint weights in the loss function
        const
    {
        if(!m_workAllocated)
        {
            unsigned int N = Y.size();
            m_Z.resize(m_H, N);
            m_output.resize(N);
            m_residual.resize(N);
            m_sqResidual.resize(N);
        }
        this->Evaluate(m_output, m_Z, X);
        utilities::VectorDiff(m_residual, Y, m_output);
        utilities::VectorHadamard(m_sqResidual, 1.0, m_residual, m_residual);
        double lossFunction = 0.0
        lossFunction += utilities::VectorDot(lfWeights.residuals, m_sqResidual);
        lossFunction += lfWeights.l1Alpha*utilities::MatrixL1(m_alpha);
        lossFunction += lfWeights.l1Beta*utilities::VectorL1(m_beta);
        lossFunction += lfWeights.l2Alpha*utilities::MatrixL2(m_alpha, m_alpha);
        lossFunction += lfWeights.l2Beta*utilities::VectorL2(m_beta, m_beta);
        return lossFunction;
    }
    
    //!
    //! Overload for the gradient of square loss function given
    //! a vector of non zero network weights, and to extract non-zero
    //! gradients of those weights
    //!
    void SingleLayerPerceptron::EvaluateSquaredLossGradient(
        dvec& nzGradients,                      //!<    Non-zero network gradients
        const dvec& nzWeights,                  //!<    Non-zero network weights
        const dvec& Y,                          //!<    Vector of N training outputs
        const utilities::matrix<double>& X,     //!<    P by N array of inputs
        const LossFunctionWeights& lfWeights)   //!<    Constraint weights in the loss function
    {
        this->UpdateNzWeights(nzWeights);
        this->EvaluateSquaredLossGradient(Y, X, lfWeights);
        this->ExtractNzGradients(nzGradients);
    }
    
    //!
    //! Evaluate the gradient of the squared loss function
    //!
    void SingleLayerPerceptron::EvaluateSquaredLossGradient(
        const dvec& Y,                          //!<    Vector of N training outputs
        const utilities::matrix<double>& X,     //!<    P by N array of inputs
        const LossFunctionWeights& lfWeights)   //!<    Constraint weights in the loss function
    {
        if(!m_workAllocated)
        {
            unsigned int N = Y.size();
            m_Z.resize(m_H, N);
            m_output.resize(N);
            m_residual.resize(N);
            m_delta.resize(N);
            m_activationDeriv.resize(m_H, N);
            m_S.resize(m_H, N);
            m_alphaGradient.resize(m_H, P);
            m_betaGradient.resize(m_H);
            m_sgnAlpha.resize(m_H, P);
            m_sgnBeta.resize(m_H);
        }
        //  Compute delta = -2*residualWeights*(y - output)*outputDeriv
        this->ActivationFunction(m_Z, X);
        this->OutputFunction(m_output, m_Z);
        this->OutputFunctionDeriv(m_outputDeriv, m_Z);
        utilities::VectorDiff(m_residual, Y, m_output);
        utilities::VectorHadamard(m_delta, 1.0, m_residual, m_outputDeriv);
        utilities::VectorHadamard(m_delta, -2.0, m_delta, lfWeights.residuals);
        //  Compute S = activationDeriv*OuterProduct(beta, delta)
        this->ActivationFunctionDeriv(m_activationDeriv, X);
        utilities::SetToConstantMatrix(m_S, 0.0);
        utilities::OuterProductIncrement(m_S, 1.0, m_beta, m_delta);
        utilities::MatrixHadamard(m_S, 1.0, m_activationDeriv, m_S);
        //  Compute alpha_gradient = S.X^T + 2*l2*alpha + l1*sgn(alpha)
        utilities::MatrixMatrixMultiply(m_alphaGradient, m_S, X, "NT");
        utilities::MatrixIncrement(m_alphaGradient, 2*lfWeights.l2Alpha, m_alpha);
        utilities::MatrixSgn(m_sgnAlpha, lfWeights.l1Alpha, m_alpha);
        utilities::MatrixIncrement(m_alphaGradient, lfWeights.l1Alpha, m_sgnAlpha);
        //  Compute beta_gradient = Z.delta + 2*l2*beta + l1*sgn(beta)
        utilities::MatrixVectorMultiply(m_betaGradient, 1.0, m_Z, m_delta);
        utilities::VectorIncrement(m_betaGradient, 2*lfWeights.l2Beta, m_beta);
        utilities::VectorSgn(m_sgnBeta, m_beta);
        utilities::VectorIncrement(m_betaGradient, lfWeights.l1Beta, m_sgnBeta);
    }

    //!
    //! Train the network using a quasi-Newton method to optimize the non-zero weights
    //!
    void SingleLayerPerceptron::Train(
        dvec& Y,                                //!<    Vector of N training outputs
        utilities::matrix<double>& X,           //!<    P by N array of training inputs
        const LossFunctionWeights& lfWeights,   //!<    Constraint weights in the loss function
        unsigned int maxIter,                   //!<    Maximum number of allowed iteractions of
                                                //!     optimization
        double tol)                             //!<    Termination tolerance of optimization
    {
        if(this->CheckDimensions(Y, X))
        {
            return;
        }
        dvec x(this->nnzWeights());
        dvec grad(this->nnzWeights());
        this->ExtractNzWeights(x);
        this->ExtractNzGradients(grad);
        utilities::optimize::bfgs(x, grad, 
            std::bind(this->EvaluateSquaredLoss, std::placeholders::1_, Y, X, lfWeights),
            std::bind(this->EvaluateSquaredLossGradient, std::placeholders::1_, 
                      std::placeholders::2_, Y, X, lfWeights), 
            maxIter, tol);                          
        this->UpdateNzWeights(x);
    }
}   //  End namespace ann
