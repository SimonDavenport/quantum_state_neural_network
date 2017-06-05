////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     This file contains an implementation of the L-BFGS quasi-Newton
//!     optimization algorithm (Limited-memory Broyden-Fletcher-Goldfarb-Shanno). 
//!     See Wikipedia article for nomenclature
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

#include "lbfgs.hpp"

namespace utilities
{
    namespace optimize
    {
        //!
        //! Default constructor
        //!
        LBFGS::LBFGS()
            :
            m_M(12),
            m_workAllocated(false)
        {}
        
        //!
        //! Set the number of iterations used to compute
        //! the Hessian inverse approximant
        //!
        void LBFGS::SetUpdateNumber(const unsigned int M)
        {
            m_M = M;
        }

        //!
        //! Allocate working memory for the l-bfgs algorithm
        //!
        void LBFGS::AllocateWork(
            const unsigned int N)   //!<    Parameter number
        {
            dvec footprint(N);
            m_prevDeltaX.resize(m_M, footprint);
            m_prevDeltaGrad.resize(m_M, footprint);
            m_prevNorm.resize(m_M);
            m_a.resize(m_M);
            m_nextGrad.resize(N);
            m_searchDir.resize(N);
            m_deltaX.resize(N);
            m_deltaGrad.resize(N);
            m_work.resize(N);
            m_workAllocated = true;
        }

        //! 
        //! Minimization of a scalar function of one or more variables
        //! using the LBFGS algorithm. 
        //!
        void LBFGS::Optimize(
            dvec& x, 
            dvec& grad,
            std::function<double(const dvec&)>& EvaluateLoss,
            std::function<void(dvec&, const dvec&)>& EvaluateGradients,
            const unsigned int maxIter, 
            const double gradTol)
        {
            if(!m_workAllocated || (m_nextGrad.size() != grad.size()))
            {
                this->AllocateWork(x.size());
            }
            int iter = 0;
            double alpha = 0.0;
            double prevLoss = EvaluateLoss(x);
            double H = 1.0; //  H represents a constant diagonal matrix
            EvaluateGradients(grad, x);
            while((VectorL2(grad) > gradTol) && (iter < (int)maxIter))
            {
                // Update search direction
                const int bound = std::min(iter, m_M);
                m_work = grad;
                for(int i = bound-1; i >= 0; --i)
                {
                    m_a[i] = VectorDot(m_prevDeltaX[i], m_work) / m_prevNorm[i];
                    VectorIncrement(m_work, -m_a[i], m_prevDeltaGrad[i]);
                }
                VectorScale(m_searchDir, -H, m_work);
                for(int i = 0; i < bound; ++i)
                {
                    double b = VectorDot(m_prevDeltaGrad[i], m_searchDir) / m_prevNorm[i];
                    VectorIncrement(m_searchDir, -m_a[i]-b, m_prevDeltaX[i]);
                }
                bool success = LineSearch(alpha, m_nextGrad, prevLoss,
                                          m_searchDir, x, grad, m_work, 
                                          EvaluateLoss, EvaluateGradients);
                if(!success)
                {
                    utilities::cout.DebuggingInfo() << "\tL-BFGS terminating early "
                                                    << "at iteration " << iter << std::endl;
                    break;
                }
                prevLoss = EvaluateLoss(x);
                //  Compute parameter and gradient increments
                VectorScale(m_deltaX, alpha, m_searchDir);
                VectorDiff(m_deltaGrad, m_nextGrad, grad);
                VectorIncrement(x, 1.0, m_deltaX);
                VectorIncrement(grad, 1.0, m_deltaGrad);
                //  Add to update history
                m_prevDeltaX.push(m_deltaX);
                m_prevDeltaGrad.push(m_deltaGrad);
                //  Compute norm and add to update history
                double norm = VectorDot(m_deltaX, m_deltaGrad);
                if(norm < 1e-20)
                {
                    utilities::cout.DebuggingInfo() << "\tWARNING: norm of delta vector below"
                                                    << " limit 1e-20" <<std::endl;
                    utilities::cout.DebuggingInfo() << "\tL-BFGS terminating early at iteration " 
                                                    << iter << std::endl;
                    break;
                }
                if(0 == iter)
                {
                    H = 1.0 / norm * VectorL2(m_deltaGrad);
                }
                m_prevNorm.push(norm);
                ++iter;
            }
            m_prevDeltaX.reset();
            m_prevDeltaGrad.reset();
            m_prevNorm.reset();
        }
    }   //  End namespace optimize
}   //  End namespace utilities
