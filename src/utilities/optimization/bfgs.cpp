////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains an implementation of the BFGS quasi-Newton
//!     optimization algorithm (Broyden-Fletcher-Goldfarb-Shanno)
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

#include "bfgs.hpp"

namespace utilities
{
    namespace optimize
    {
        //!
        //! Default constructor
        //!
        BFGS()
            :
            m_workAllocated = false
        {}

        //!
        //! Allocate working memory for the bfgs algorithm
        //!
        void BFGS::AllocateWork(
            const unsigned int N)   //!<    Parameter number
        {
            m_nextGrad.resize(N);
            m_Binv.resize(N, N);
            m_searchDir.resize(N);
            m_deltaX.resize(N);
            m_deltaGrad.resize(N);
            m_work.resize(N);
            m_workAllocated = true;
        }

        //! 
        //! Minimization of a scalar function of one or more variables
        //! using the BFGS algorithm. Implementation trasnlated from
        //! scipy.optimize._minimize_bfgs.
        //!
        void BFGS::Optimize(
            dvec& x, 
            dvec& grad,
            std::function<double(const dvec&)> EvaluateLoss,
            std::function<void(const dvec&, const dvec&)> EvaluateGradients,
            const unsigned int maxIter, 
            const double gradTol)
        {
            if(!m_workAllocated)
            {
                this->AllocateWork(N);
            }
            SetToIdentityMatrix(m_Binv);
            unsigned int iter = 0;
            double alpha = 0.0;
            double prevLoss = EvaluateLoss(x);
            while((VectorL2(grad) > gradTol) && (iter < maxIter))
            {
                SymmetricMatrixVectorMultiply(m_searchDir, -1.0, m_Binv, grad);
                bool success = LineSearch(alpha, m_searchDir, x, grad, m_nextGrad, m_work,
                                          prevLoss, EvaluateLoss, EvaluateGradients);
                if(!success)    break;
                //  Compute parameter and gradient increments
                VectorScale(m_deltaX, alpha, m_searchDir);
                VectorDiff(m_deltaGrad, m_nextGrad, grad);
                VectorIncrement(x, 1.0, m_deltaX);
                VectorIncrement(grad, 1.0, m_deltaGrad);
                //  Update hessian inverse approximant via the
                //  Sherman-Morrison formula:
                //  Binv_{k+1} = Binv_k + (alpha + y^T.v) s.s^T / norm^2
                //  - v.s^T/norm - s.v^T/norm
                double norm = std::max(VectorDot(m_deltaX, m_deltaGrad), 0.001);
                SymmetricMatrixVectorMultiply(m_work, 1.0, m_Binv, m_searchDir);
                SymmetricOuterProductIncrement(m_Binv, (norm+VectorDot(m_deltaGrad, m_work))/
                                               (norm*norm), m_deltaX);
                SymmetricOuterProductIncrement(m_Binv, -1.0/norm, m_work, m_deltaX);
                prevLoss = EvaluateLoss(x);
                ++iter;
            }
        }
    }   //  End namespace optimize
}   //  End namespace utilities
