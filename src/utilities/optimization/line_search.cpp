////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     This file contains functions to perform a line search optimization. 
//!     The implementation has been lifted from scipy.optimize.linesearch, 
//!     where it is called scalar_search_wolfe2. Some notation is changed
//!     for clarity.
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

#include "line_search.hpp"

namespace utilities
{
    namespace optimize
    {
        //!
        //! Implementation of a Wolfe line search algorithm. Determines 
        //! a scale parameter alpha satisfying the strong Wolfe conditions. 
        //! Implementation lifted from scalar_search_wolfe2 in 
        //! scipy.optimize.linesearch.
        //!
        bool LineSearch(
            double& alpha,          //!< Optimial scale factor to be set
            dvec& nextGrad,         //!< Gradient at the optimal point to be set
            const double prevLoss,  //!< Current value of the loss function
            const dvec& searchDir,  //!< Search direction
            const dvec& x,          //!< Current value of the loss function arguments
            const dvec& grad,       //!< Current value of the gradient
            dvec& work,             //!< Working memory space
            std::function<double(const dvec&)>& EvaluateLoss,
            std::function<void(dvec&, const dvec&)>& EvaluateGradients)
        {
            double startLoss = EvaluateLoss(x);
            double startDeriv = VectorDot(grad, searchDir);
            double optLoss = startLoss;
            double optDeriv = startDeriv;
            // Set initial search limits
            double alpha0 = 0.0;
            double alpha1 = std::min(1.0, 2.02*(startLoss - prevLoss)/startDeriv);
            if(alpha1 <= 0.0)
            {
                alpha1 = 1.0;
            }
            double alpha0Loss = startLoss;
            double alpha0Deriv = startDeriv;
            double alpha1Loss = LossIncrement(x, alpha1, searchDir, work, EvaluateLoss);
            //  Iterate search region
            bool success = false;
            for(unsigned int iter=0; iter<maxIter; ++iter)
            {
                if(alpha1 == 0.0)
                {
                    success = true;
                    break;
                }
                //  Check Wolfe conditions
                if((alpha1Loss > startLoss + c1 * alpha1 * startDeriv) ||
                   ((alpha1Loss >= alpha0Loss) && (iter > 0)))
                {
                    success = Interpolate(alpha, optLoss, optDeriv, nextGrad, alpha0, alpha1, 
                         alpha0Loss, alpha1Loss, alpha0Deriv, startLoss, startDeriv, 
                         searchDir, x, work, EvaluateLoss, EvaluateGradients);
                    break;
                }
                GradIncrement(nextGrad, x, alpha1, searchDir, work, EvaluateGradients);
                double alpha1Deriv = VectorDot(nextGrad, searchDir);
                //  Check Wolfe conditions
                if(std::abs(alpha1Deriv) <= -c2*alpha0Deriv)
                {
                    alpha = alpha1;
                    success = true;
                    break;
                }
                if(alpha1Deriv >= 0)
                {
                    success = Interpolate(alpha, optLoss, optDeriv, nextGrad, alpha1, alpha0, 
                         alpha1Loss, alpha0Loss, alpha1Deriv, startLoss, startDeriv, 
                         searchDir, x, work, EvaluateLoss, EvaluateGradients);
                    break;
                }
                //  Iterate search region
                alpha0 = alpha1;
                alpha1 = 2.0*alpha0;
                alpha0Loss = alpha1Loss;
                alpha1Loss = LossIncrement(x, alpha1, searchDir, work, EvaluateLoss);
                alpha0Deriv = alpha1Deriv;
            }
            if(!success)
            {
                alpha = alpha1;
                std::cerr << "\tWARNING: Line search algorithm failed to converge after ";
                std::cerr << maxIter << " iterations " << std::endl;
            }
            return success;
        }

        //!
        //! Get the loss function for parameters x1 = x + alpha*searchDir
        //!
        double LossIncrement(
            const dvec& x,          //!< Current loss function arguments
            const double& alpha,    //!< Scale factor
            const dvec& searchDir,  //!< Search direction
            dvec& x1,               //!< New arguments to be set
            std::function<double(const dvec&)>& EvaluateLoss)
        {
            x1 = x;
            VectorIncrement(x1, alpha, searchDir);
            return EvaluateLoss(x1);
        }
      
        //!
        //! Get gradient of loss function for parameters x1 = x + alpha*searchDir
        //!
        void GradIncrement(
            dvec& newGrad,          //!< New gradient value to be set
            const dvec& x,          //!< Loss function arguments
            const double& alpha,    //!< Scale factor
            const dvec& searchDir,  //!< Direction of search
            dvec& x1,               //!< New arguments to be set
            std::function<void(dvec&, const dvec&)>& EvaluateGradients)
        {
            x1 = x;
            VectorIncrement(x1, alpha, searchDir);
            EvaluateGradients(newGrad, x1);
        }
        
        //!
        //! Subroutine of the linear search algorithm, lifted from the
        //! _zoom function in scipy.optimize.linesearch. Interpolates to find 
        //! a trial step length between alpha0 and alpha1. A cubic scheme is
        //! used; then if the result is too close to the end points, then 
        //! use quadratic interpolation, then if again it is too close, 
        //! bisection is used. 
        //!
        bool Interpolate(
            double& alpha,          //!< Optimal interpolated scale factor to be set
            double& optLoss,        //!< Optimal interpolated loss function to be set
            double& optDeriv,       //!< Optimal loss function deriv to be set
            dvec& nextGrad,         //!< Gradient at the optimal point to be set
            double& alpha0,         //!< Scale factor at lower edge of search region
            double& alpha1,         //!< Scale factor at upper edge of search region
            double& alpha0Loss,     //!< Loss function at lower edge of search region
            double& alpha1Loss,     //!< Loss function at upper edge of search region
            double& alpha0Deriv,    //!< Loss function deriv at lower edge of region
            const double& startLoss, //!< Initial loss function value
            const double& startDeriv,//!< Initial loss funtion deriv
            const dvec& searchDir,  //!< Search direction
            const dvec& x,          //!< Current loss function arguments
            dvec& work,             //!< Working memory space
            std::function<double(const dvec&)>& EvaluateLoss,
            std::function<void(dvec&, const dvec&)>& EvaluateGradients)
        {
            double alphaRec = 0.0;
            double lossRec = 0.0;
            double dAlpha = 0.0;
            double lowerLim = 0.0;
            double upperLim = 0.0;
            bool success = false;
            for(unsigned int iter=0; iter<maxIter; ++iter)
            {
                dAlpha = alpha1 - alpha0;
                if(dAlpha < 0)
                {
                    lowerLim = alpha1;
                    upperLim = alpha0;
                }
                else
                {
                    lowerLim = alpha0;
                    upperLim = alpha1;
                }
                //  Parameters delta1 and delta2 are fixed in the header.
                double cubicCheck = delta1 * dAlpha;
                double quadCheck = delta2 * dAlpha;
                if(iter > 0)
                {
                    //  Cubic interpolation
                    alpha = CubicMin(alpha0, alpha0Loss, alpha0Deriv, 
                                     alpha1, alpha1Loss, alphaRec, lossRec);
                }
                if((0 == iter) || (alpha > upperLim - cubicCheck) || (alpha < lowerLim + cubicCheck))
                {
                    //  Quadratic interpolation
                    alpha = QuadMin(alpha0, alpha0Loss, alpha0Deriv, 
                                    alpha1, alpha1Loss);
                    if((alpha > upperLim - quadCheck) || (alpha < lowerLim + quadCheck))
                    {
                        //  Linear interpolation
                        alpha = alpha0 + 0.5*dAlpha;
                    }
                }
                //  Check Wolfe conditions
                optLoss = LossIncrement(x, alpha, searchDir, work, EvaluateLoss);
                if((optLoss > startLoss + c1*alpha*startDeriv) || optLoss > alpha0Loss)
                {
                    lossRec = alpha1Loss;
                    alphaRec = alpha1;
                    alpha1 = alpha;
                    alpha1Loss = optLoss;
                }
                else
                {
                    GradIncrement(nextGrad, x, alpha, searchDir, work, EvaluateGradients);
                    optDeriv = VectorDot(nextGrad, searchDir);
                    if(std::abs(optDeriv) <= -c2*startDeriv)
                    {
                        success = true;
                        break;
                    }
                    if(optDeriv*(alpha1 - alpha0) >= 0)
                    {
                        lossRec = alpha1Loss;
                        alphaRec = alpha1;
                        alpha1 = alpha0;
                        alpha1Loss = alpha0Loss;
                    }
                    else
                    {
                        lossRec = alpha0Loss;
                        alphaRec = alpha0;
                    }
                    alpha0 = alpha;
                    alpha0Loss = optLoss;
                    alpha0Deriv = optDeriv;
                }
            }
            return success;
        }
        
        //!
        //! Find the minimizer for a cubic polynomial that goes through 
        //! the points (a, fa), (b, fb) and (c,fc) with derivative fpa
        //! at a. Implementation lifted from scipy.optimize.linesearch
        //! where it is the _cubicmin function.
        //!
        double CubicMin(
            const double a,
            const double fa,
            const double fpa, 
            const double b,
            const double fb, 
            const double c,
            const double fc)
        {
            double C = fpa;
            double db = b - a;
            double dc = c - a;
            double denom = (db * dc)*(db * dc)*(db - dc);
            double A = dc * dc * (fb - fa - C * db) - db * db * (fc - fa - C * dc);
            double B = -dc * dc * dc * (fb - fa - C * db) + db * db * db * (fc - fa - C * dc);
            A /= denom;
            B /= denom;
            return a + (-B + sqrt(B * B - 3 * A *C)) / (3 * A);
        }

        //!
        //! Find the minimizer for a quadratic polynomial that goes through
        //! the points (a,fa), (b,fb) with derivative fpa at a. Implementation
        //! lifted from scipy.optimize.linesearch where it is the _quadmin
        //! function.
        //!
        double QuadMin(
            const double a,
            const double fa,
            const double fpa,
            const double b,
            const double fb)
        {
            double D = fa;
            double C = fpa;
            double db = b - a * 1.0;
            double B = (fb - D - C * db) / (db *db);
            return a - C / (2.0 * B);
        }
    }   //  End namespace optimize
}   //  End namespace utilities
