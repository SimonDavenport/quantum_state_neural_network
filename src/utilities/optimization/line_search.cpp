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

#include "line_search.hpp"

namespace utilities
{
    namespace optimize
    {
        //!
        //! Implementation of a Wolfe line search algorithm. Determines 
        //! a scale parameter alpha satisfying the strong Wolfe conditions
        //!
        bool LineSearch(
            double& alpha, 
            dvec& nextGrad,
            const double prevLoss,
            const dvec& searchDir,
            const dvec& x, 
            const dvec& grad,
            dvec& work,
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
            //  Set initial loss function values and derivs at the search region limits
            double alpha0Loss = startLoss;
            double alpha0Deriv = startDeriv;
            double alpha1Loss = LossIncrement(x, alpha1, searchDir, work, EvaluateLoss);
            double alpha1Deriv = 0.0;
            //  Iterate search region
            for(unsigned int i=0; i<maxIter; ++i)
            {
                if(alpha1 < 0)
                {
                    break;
                }
                //  Check Wolfe conditions
                if((alpha1Loss > startLoss + c1*alpha1*startDeriv) ||
                   ((alpha1Loss >= alpha0Loss) && (i>0)))
                {
                    Zoom(alpha, optLoss, optDeriv, nextGrad, alpha0, alpha1, 
                         alpha0Loss, alpha1Loss, alpha0Deriv, startLoss, startDeriv, 
                         searchDir, x, work, EvaluateLoss, EvaluateGradients);
                    break;
                }
                GradIncrement(nextGrad, x, alpha1, searchDir, work, EvaluateGradients);
                alpha1Deriv = VectorDot(nextGrad, searchDir);
                //  Check Wolfe conditions
                if(std::abs(alpha1Deriv) <= -c2*alpha0Deriv)
                {
                    alpha = alpha1;
                    optLoss = alpha1Loss;
                    optDeriv = alpha1Deriv;
                    break;
                }
                if(alpha1Deriv >= 0)
                {
                    Zoom(alpha, optLoss, optDeriv, nextGrad, alpha1, alpha0, 
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
            return true;
        }

        //!
        //! Get the loss function for parameters x1 = x + alpha*searchDir
        //!
        double LossIncrement(
            const dvec& x,
            const double& alpha, 
            const dvec& searchDir,
            dvec& x1,
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
            dvec& newGrad,
            const dvec& x,
            const double& alpha, 
            const dvec& searchDir,
            dvec& x1,
            std::function<void(dvec&, const dvec&)>& EvaluateGradients)
        {
            x1 = x;
            VectorIncrement(x1, alpha, searchDir);
            EvaluateGradients(newGrad, x1);
        }
        
        //!
        //! Subroutine of the linear search algorithm
        //!
        void Zoom(
            double& alpha,
            double& optLoss,
            double& optDeriv,
            dvec& nextGrad,
            double& alpha0, 
            double& alpha1, 
            double& alpha0Loss, 
            double& alpha1Loss, 
            double& alpha0Deriv,
            const double& startLoss, 
            const double& startDeriv,
            const dvec& searchDir,
            const dvec& x, 
            dvec& work,
            std::function<double(const dvec&)>& EvaluateLoss,
            std::function<void(dvec&, const dvec&)>& EvaluateGradients)
        {
            double alphaRec = 0.0;
            double lossRec = 0.0;
            double dAlpha = 0.0;
            double lowerLim = 0.0;
            double upperLim = 0.0;
            for(unsigned int i=0; i<maxIter; ++i)
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
                double cubicCheck = delta1 * dAlpha;
                double quadCheck = delta2 * dAlpha;
                if(i > 0)
                {
                    alpha = CubicMin(alpha0, alpha0Loss, alpha0Deriv, 
                                     alpha1, alpha1Loss, alphaRec, lossRec);
                }
                if((0 == i) || (alpha > upperLim - cubicCheck) || (alpha < lowerLim + cubicCheck))
                {
                    alpha = QuadMin(alpha0, alpha0Loss, alpha0Deriv, 
                                    alpha1, alpha1Loss);
                    if((alpha > upperLim - quadCheck) || (alpha < lowerLim + quadCheck))
                    {
                        alpha = alpha0 + 0.5*dAlpha;
                    }
                }
                optLoss = LossIncrement(x, alpha, searchDir, work, EvaluateLoss);
                //  Check Wolfe conditions
                if((optLoss > startLoss + c1*alpha*startDeriv) || optLoss >= alpha0Loss)
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
        }
        
        //!
        //! Find the minimizer for a cubic polynomial that goes through 
        //! the points (a, fa), (b, fb) and (c,fc) with derivative fpa
        //! at a. 
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
        //! the points (a,fa), (b,fb) with derivative fpa at a.
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
