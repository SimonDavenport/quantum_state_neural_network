////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     This file contains an implementation of a single-layer perceptron
//!     neural network. Note: biases in the activation function argument
//!     can be included via a constant feature vector. 
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

namespace ann
{   
    //!
    //! Implementation of activation function, mapping N inputs X of size  
    //! P to H hidden node parameters Z via weights alpha.
    //!
    void SingleLayerPerceptron::ActivationFunction(
        utilities::matrix<double>& Z,       //!<    H by N hidden 
                                            //!     node parameters
        const utilities::matrix<double>& X) //!<    P by N inputs
    {
        utilities::MatrixMatrixMultiply(Z, m_alpha, X, "NN");
        for(auto& it : Z.m_data)
        {
            it = m_ActivationImpl(it);
        }
    }

    //!
    //! Implementation of activation function derivative
    //!
    void SingleLayerPerceptron::ActivationFunctionDeriv(
        utilities::matrix<double>& Z,       //!<    H by N hidden 
                                            //!     node parameters
        const utilities::matrix<double>& X) //!<    P by N inputs
    {
        utilities::MatrixMatrixMultiply(Z, m_alpha, X, "NN");
        for(auto& it : Z.m_data)
        {
            it = m_ActivationDerivImpl(it);
        }
    }
    
    //!
    //! Implementation of the output function, mapping hidden node 
    //! parameters Z to output parameters Y via weights beta
    //!
    void SingleLayerPerceptron::OutputFunction(
        dvec& Y,                            //!<    Vector of N outputs
        const utilities::matrix<double>& Z) //!<    H by N hidden node parameters
    {
        utilities::MatrixVectorMultiply(Y, 1.0, Z, m_beta, 'T');
        for(auto& it : Y)
        {
            it = m_OutputFunctionImpl(m_betaBias + it);
        }
    }
    
    //!
    //! Implementation of the output function derivative
    //!
    void SingleLayerPerceptron::OutputFunctionDeriv(
        dvec& Y,                            //!<    Vector of N outputs
        const utilities::matrix<double>& Z) //!<    H by N hidden 
                                            //!     node parameters
    {
        utilities::MatrixVectorMultiply(Y, 1.0, Z, m_beta, 'T');
        for(auto& it : Y)
        {
            it = m_OutputFunctionDerivImpl(m_betaBias + it);
        }
    }        
    
    //!
    //! Default constructor
    //!
    SingleLayerPerceptron::SingleLayerPerceptron()
        :
        m_includeBetaBias(true),
        m_P(0),
        m_H(0),
        m_N(0),
        m_workAllocated(false),
        m_ActivationImpl(Logistic),
        m_ActivationDerivImpl(LogisticDeriv),
        m_OutputFunctionImpl(Unit),
        m_OutputFunctionDerivImpl(UnitDeriv)
    {}
    
    //!
    //! Constructor for a certian number of features and hidden nodes
    //!
    SingleLayerPerceptron::SingleLayerPerceptron(
        const unsigned int P,       //!<    Number of features
        const unsigned int H)       //!<    Number of hidden nodes in layer
        :
        m_includeBetaBias(true),
        m_P(P),
        m_H(H),
        m_N(0),
        m_workAllocated(false),
        m_ActivationImpl(Logistic),
        m_ActivationDerivImpl(LogisticDeriv),
        m_OutputFunctionImpl(Unit),
        m_OutputFunctionDerivImpl(UnitDeriv)
    {
        m_alpha.resize(m_H, m_P);
        m_beta.resize(m_H);
    }

    //! 
    //! Destructor
    //!
    SingleLayerPerceptron::~SingleLayerPerceptron()
    {}

    //!
    //! Allocate working space
    //!
    void SingleLayerPerceptron::AllocateWork(
        unsigned int N)     //!<    Working space to allocate for
    {
        if(!m_workAllocated || (m_N != N))
        {
            m_N = N;
            m_Z.resize(m_H, m_N);
            m_outputDeriv.resize(m_N);
            m_output.resize(m_N);
            m_residual.resize(m_N);
            m_sqResidual.resize(m_N);
            m_delta.resize(m_N);
            m_activationDeriv.resize(m_H, m_N);
            m_S.resize(m_H, m_N);
            m_alphaGradient.resize(m_H, m_P);
            m_betaGradient.resize(m_H);
            m_sgnAlpha.resize(m_H, m_P);
            m_sgnBeta.resize(m_H);
            m_workAllocated = true;
        }
    }
    
    //!
    //! Check the dimensions of the input containers
    //!
    bool SingleLayerPerceptron::CheckDimensions(
        const dvec& Y,                      //!<    Vector of N outputs
        const utilities::matrix<double>& X) //!<    P by N array of inputs
    {
        if(X.m_dSecond != Y.size())
        {
            std::cerr << "ERROR: X and Y dimension mismatch " << std::endl;
            return true;
        }
        if(X.m_dLeading != m_P)
        {
            std::cerr << "ERROR: X feature dimension mismatch " << std::endl;
            return true;
        }
        if(Y.size() != m_N)
        {
            std::cerr << "ERROR: dimension mismatch with work allocation " << std::endl;
            return true;
        }
        return false;
    }

    //!
    //! Get total number of alpha weights
    //!
    unsigned int SingleLayerPerceptron::countAlphaWeights() const
    {
        return m_H * m_P;
    }
   
    //!
    //! Get total number of weights
    //!
    unsigned int SingleLayerPerceptron::countWeights() const
    {
        return this->countAlphaWeights() + m_H;
    }
    
    //!
    //! Function to set the current weights and the beta bias to be random
    //!
    void SingleLayerPerceptron::RandomizeWeights(
        const double scale,         //!<    Scale of weights
        const unsigned int seed)    //!<    Random seed
    {
        dvec tempRandom(this->countWeights() + 1);
        utilities::SetToRandomVector(tempRandom, scale, seed);
        utilities::ToSubVector(m_alpha.m_data, tempRandom, 0);
        utilities::ToSubVector(m_beta, tempRandom, this->countAlphaWeights());
        m_betaBias = tempRandom[this->countWeights()];
    }

    //!
    //! Set the activation function
    //!
    void SingleLayerPerceptron::SetActivationFunction(
        std::function<double(const double& x)> activationImpl,
                            //!< Implementaion of the activation function
        std::function<double(const double& x)> activationDerivImpl)
                            //!< Implementation of activation func deriv
    
    {
        m_ActivationImpl = activationImpl;
        m_ActivationDerivImpl = activationDerivImpl;
    }
    
    //!
    //! Set the loss function weights
    //!
    void SingleLayerPerceptron::SetLossFunctionWeights(
        const LossFunctionWeights& lfWeights)
    {
        m_lfWeights = lfWeights;
    }
    
    //!
    //! Set weights (note that the mask is imposed on the set weights)
    //!
    void SingleLayerPerceptron::SetWeights(
        const utilities::matrix<double>& alpha, //!<    Updated alpha weights
        const dvec& beta)                       //!<    Updated beta weights
        
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
        utilities::matrix<double>& alpha,   //!<    Alpha weights container
        dvec& beta)                         //!<    Beta weights container
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
        zeros = m_zeros;
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
        return this->countAlphaWeights() - m_zeros.size();
    }

    //!
    //! Update non-zero weights from a vector container
    //!
    void SingleLayerPerceptron::SetNzWeights(
        const dvec& nzWeights)      //! Vector container for non-zero 
                                    //! alpha and beta weights
    {
        utilities::ToSubVector(m_alpha.m_data, nzWeights, 0, m_zeros);
        utilities::ToSubVector(m_beta, nzWeights, this->nnzAlpha());
        if(m_includeBetaBias)
        {
            m_betaBias = nzWeights[this->nnzWeights()];
        }
    }
    
    //!
    //! Get non-zero weights and place in a vector container
    //!
    void SingleLayerPerceptron::GetNzWeights(
        dvec& nzWeights)                //! Vector container for non-zero 
                                        //! alpha and beta weights
        const
    {
        utilities::FromSubVector(m_alpha.m_data, nzWeights, 0, m_zeros);
        utilities::FromSubVector(m_beta, nzWeights, this->nnzAlpha());
        if(m_includeBetaBias)
        {
            nzWeights[this->nnzWeights()] = m_betaBias;
        }
    }
    
    //!
    //! Get the gradients of the non-zero weights in a single vector container
    //!
    void SingleLayerPerceptron::GetNzGradients(
        dvec& gradients)                //!<  Vector container for non-zero 
                                        //!<  alpha and beta gradients
        const
    {
        utilities::FromSubVector(m_alphaGradient.m_data, gradients, 0, m_zeros);
        utilities::FromSubVector(m_betaGradient, gradients, this->nnzAlpha());
        if(m_includeBetaBias)
        {
            gradients[this->nnzWeights()] = m_betaBiasGradient;
        }
    }

    //!
    //! Get total number of non-zero beta biases
    //!
    unsigned int SingleLayerPerceptron::nnzBiases() const
    {
        return m_includeBetaBias;
    }

    //!
    //! Evalute the current output of the neural network for given inputs X,
    //! where X contains N inputs of size P features in a P by N matrix, and write
    //! it to the vector Y.
    //!
    void SingleLayerPerceptron::Evaluate(
        dvec& Y,                                //!<    Vector of N outputs
        const utilities::matrix<double>& X)     //!<    P by N array of inputs
    {
        bool workRequest = (Y.size() != m_N);
        if(workRequest)
        {
            unsigned int N = Y.size();
            m_Z.resize(m_H, N);
        }
        this->ActivationFunction(m_Z, X);
        this->OutputFunction(Y, m_Z);
        if(workRequest)
        {
            m_Z.resize(m_H, m_N);
        }
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
        const utilities::matrix<double>& X)     //!<    P by N array of training inputs
    {
        bool workRequest = (Y.size() != m_N);
        if(workRequest)
        {
            unsigned int N = Y.size();
            m_Z.resize(m_H, N);
            m_output.resize(N);
            m_residual.resize(N);
            m_sqResidual.resize(N);
        }
        this->Evaluate(m_output, X);
        utilities::VectorDiff(m_residual, Y, m_output);
        double lossFunction = 0.0;
        if(m_lfWeights.usingResidualWeights)
        {
            utilities::VectorHadamard(m_sqResidual, 1.0, m_residual, m_residual);
            lossFunction += utilities::VectorDot(m_lfWeights.residualWeights, m_sqResidual);
        }
        else
        {
            lossFunction += utilities::VectorL2(m_residual);
        }
        lossFunction += m_lfWeights.l1Alpha*utilities::MatrixL1(m_alpha);
        lossFunction += m_lfWeights.l1Beta*utilities::VectorL1(m_beta);
        lossFunction += m_lfWeights.l2Alpha*utilities::MatrixL2(m_alpha);
        lossFunction += m_lfWeights.l2Beta*utilities::VectorL2(m_beta);
        if(workRequest)
        {
            m_Z.resize(m_H, m_N);
            m_output.resize(m_N);
            m_residual.resize(m_N);
            m_sqResidual.resize(m_N);
        }
        return lossFunction;
    }
   
    //!
    //! Evaluate the gradient of the squared loss function
    //!
    void SingleLayerPerceptron::EvaluateSquaredLossGradient(
        const dvec& Y,                          //!<    Vector of N training outputs
        const utilities::matrix<double>& X)     //!<    P by N array of inputs
    {
        bool workRequest = (Y.size() != m_N);
        if(workRequest)
        {
            unsigned int N = Y.size();
            m_Z.resize(m_H, N);
            m_output.resize(N);
            m_residual.resize(N);
            m_delta.resize(N);
            m_activationDeriv.resize(m_H, N);
            m_S.resize(m_H, N);
            m_alphaGradient.resize(m_H, m_P);
        }
        if(!m_workAllocated)
        {
            m_betaGradient.resize(m_H);
            m_sgnAlpha.resize(m_H, m_P);
            m_sgnBeta.resize(m_H);
        }
        //  Compute delta = -2*residualWeights*(y - output)*outputDeriv
        this->ActivationFunction(m_Z, X);
        this->OutputFunction(m_output, m_Z);
        this->OutputFunctionDeriv(m_outputDeriv, m_Z);
        utilities::VectorDiff(m_residual, Y, m_output);
        utilities::VectorHadamard(m_delta, -2.0, m_residual, m_outputDeriv);
        if(m_lfWeights.usingResidualWeights)
        {
            utilities::VectorHadamardIncrement(m_delta, 1.0, m_lfWeights.residualWeights);
        }
        //  Compute S = activationDeriv*OuterProduct(beta, delta)
        this->ActivationFunctionDeriv(m_activationDeriv, X);
        utilities::SetToConstantMatrix(m_S, 0.0);
        utilities::OuterProductIncrement(m_S, 1.0, m_beta, m_delta);
        utilities::MatrixHadamardIncrement(m_S, 1.0, m_activationDeriv);
        //  Compute alphaGradient = S.X^T + 2*l2*alpha + l1*sgn(alpha)
        utilities::MatrixMatrixMultiply(m_alphaGradient, m_S, X, "NT");
        if(m_lfWeights.l2Alpha != 0)
        {
            utilities::MatrixIncrement(m_alphaGradient, 2*m_lfWeights.l2Alpha, m_alpha);
        }
        if(m_lfWeights.l1Alpha != 0)
        {
            utilities::MatrixSgn(m_sgnAlpha, m_alpha);
            utilities::MatrixIncrement(m_alphaGradient, m_lfWeights.l1Alpha, m_sgnAlpha);
        }
        //  Compute betaGradient = Z.delta + 2*l2*beta + l1*sgn(beta)
        utilities::MatrixVectorMultiply(m_betaGradient, 1.0, m_Z, m_delta, 'N');
        if(m_lfWeights.l2Beta != 0)
        {
            utilities::VectorIncrement(m_betaGradient, 2 * m_lfWeights.l2Beta, m_beta);
        }
        if(m_lfWeights.l1Beta != 0)
        {
            utilities::VectorSgn(m_sgnBeta, m_beta);
            utilities::VectorIncrement(m_betaGradient, m_lfWeights.l1Beta, m_sgnBeta);
        }
        // Compute betaBiasGradient = sum_i delta + 2*l2*betaBias + l1*sgn(betaBias)
        m_betaBiasGradient = utilities::VectorSum(m_delta);
        m_betaBiasGradient += 2 * m_lfWeights.l2Beta * m_betaBias;
        m_betaBiasGradient += 2 * m_lfWeights.l2Beta * ((m_betaBias > 0) - (m_betaBias < 0));
        if(workRequest)
        {
            m_Z.resize(m_H, m_N);
            m_output.resize(m_N);
            m_residual.resize(m_N);
            m_delta.resize(m_N);
            m_activationDeriv.resize(m_H, m_N);
            m_S.resize(m_H, m_N);
        }
    }
}   //  End namespace ann
