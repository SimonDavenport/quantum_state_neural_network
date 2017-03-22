////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains functions to perform a line search optimization
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

#ifndef _LINE_SEARCH_HPP_INCLUDED_
#define _LINE_SEARCH_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <functional>
#include "../linear_algebra/dense_matrix.hpp"
#include "../linear_algebra/dense_vector.hpp"
#include "../general/dvec_def.hpp"

namespace utilities
{
    namespace optimize
    {
        static const double c1 = 0.0001;  //!<    Armijo constraint weight
        static const double c2 = 0.9;     //!<    curvature constraint weight
        static const double delta1 = 0.2; //!<    Cubic interpolant check
        static const double delta2 = 0.1; //!<    Quadratic interpolant check
        static const unsigned int maxIter = 10; 
                                    //!<    Max iteration number in line searches
        bool LineSearch(double& alpha, dvec& nextGrad, const double prevLoss,
                        const dvec& searchDir, const dvec& x, const dvec& grad,
                        dvec& work, std::function<double(const dvec&)>& EvaluateLoss,
                        std::function<void(dvec&, const dvec&)>& EvaluateGradients);
        double LossIncrement(const dvec& x, const double& alpha, 
                             const dvec& searchDir, dvec& x1,
                             std::function<double(const dvec&)>& EvaluateLoss);
        void GradIncrement(dvec& newGrad, const dvec& x, const double& alpha, 
                           const dvec& searchDir, dvec& x1,
                           std::function<void(dvec&, const dvec&)>& EvaluateGradients);
        void Zoom(double& alpha, double& optLoss, double& optDeriv,
                  dvec& nextGrad, double& alpha0, double& alpha1, 
                  double& alpha0Loss, double& alpha1Loss, 
                  double& alpha0Deriv, const double& startLoss, 
                  const double& startDeriv, const dvec& searchDir, const dvec& x, 
                  dvec& work, std::function<double(const dvec&)>& EvaluateLoss,
                  std::function<void(dvec&, const dvec&)>& EvaluateGradients);
        double CubicMin(const double a, const double fa, const double fpa, 
                        const double b, const double fb, const double c, 
                        const double fc);
        double QuadMin(const double a, const double fa, const double fpa,
                       const double b, const double fb);
    }   //  End namespace optimize
}   //  End namespace utilities
#endif
