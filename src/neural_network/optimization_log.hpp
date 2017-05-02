////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		A utility to keep a log of the loss function value and other parameters 
//!     during certian network training procedures
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

#ifndef _OPTIMIZATION_LOG_HPP_INCLUDED_
#define _OPTIMIZATION_LOG_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../utilities/general/run_script.hpp"
#include "../utilities/general/dvec_def.hpp"
#include "../utilities/general/cout_tools.hpp"

//!
//! Class container for loss logs
//!
class OptimizationLog
{
    private:
    static const int _PYTHON_VERSION_ = 3;
    dvec minLog;    //!<    Log of min parameter for a set of trials
    dvec maxLog;    //!<    Log of max parameter for a set of trials
    dvec meanLog;   //!<    Log of mean parameter for a set of trials
    dvec optLog;    //!<    Log of refined optimal parameter
    public:
    void Record(const dvec& parameters, const double optParameter);
    void Plot(const bool takeLog, const std::string figName, 
              const std::string label, const std::string paramName,
              const std::vector<unsigned int >& dependent);
};
#endif
