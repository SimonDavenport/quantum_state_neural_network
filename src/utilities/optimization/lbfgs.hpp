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

#ifndef _LBFGS_HPP_INCLUDED_
#define _LBFGS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <functional>
#include <iostream>
#include "../data_structures/fixed_size_queue.hpp"
#include "../general/dvec_def.hpp"
#include "../linear_algebra/dense_vector.hpp"
#include "line_search.hpp"
#if _DEBUG_
#include "../general/debug.hpp"
#endif

namespace utilities
{
    namespace optimize
    {
        class LBFGS
        {
            private:
            FixedSizeQueue<dvec> m_prevDeltaX;   //!<    Queue previous deltaX
            FixedSizeQueue<dvec> m_prevDeltaGrad;//!<    Queue previous deltaGrad
            FixedSizeQueue<double> m_prevNorm;   //!<    Queue previous norm
            int m_M;                    //!<    Number of previous updates 
                                        //!     to compute hessian approx
            dvec m_a;                   //!<    Working space used in Hessian computation
            dvec m_nextGrad;            //!<    Gradient at next iteration
            dvec m_searchDir;           //!<    Conjugate vector
            dvec m_deltaX;              //!<    Change in parameters at iteration
            dvec m_deltaGrad;           //!<    Change in gradient at iteration
            dvec m_work;                //!<    Working space
            bool m_workAllocated;       //!<    Flag set when working memory allocated
            public:
            LBFGS();
            void SetUpdateNumber(const unsigned int M);
            void AllocateWork(const unsigned int N);
            void Optimize(dvec& x, dvec& grad, 
                          std::function<double(const dvec&)>& EvaluateLoss,
                          std::function<void(dvec&, const dvec&)>& EvaluateGradients,
                          const unsigned int maxIter, const double gradTol);
        };
    }   //  End namespace optimize
}   //  End namespace utilities
#endif
