////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains an implementation of the BFGS quasi-Newton
//!     optimization algorithm (Broyden-Fletcher-Goldfarb-Shanno). 
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

#ifndef _BFGS_HPP_INCLUDED_
#define _BFGS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <functional>
#include "../general/dvec_def.hpp"
#include "../linear_algebra/dense_matrix.hpp"
#include "../linear_algebra/dense_vector.hpp"
#include "line_search.hpp"
#if _DEBUG_
#include "../general/debug.hpp"
#endif

namespace utilities
{
    namespace optimize
    {
        class BFGS
        {
            private:
            dvec m_nextGrad;            //!<    Gradient at next iteration
            matrix<double> m_Binv;      //!<    Approximant to Hessian inverse
            dvec m_searchDir;           //!<    Conjugate vector
            dvec m_deltaX;              //!<    Change in parameters at iteration
            dvec m_deltaGrad;           //!<    Change in gradient at iteration
            dvec m_work;                //!<    Working space
            bool m_workAllocated;       //!<    Flag set when working memory allocated
            public:
            BFGS();
            void AllocateWork(const unsigned int N);
            void Optimize(dvec& x, dvec& grad, 
                          std::function<double(const dvec&)>& EvaluateLoss,
                          std::function<void(dvec&, const dvec&)>& EvaluateGradients,
                          const unsigned int maxIter, const double gradTol);
        };
    }   //  End namespace optimize
}   //  End namespace utilities
#endif
