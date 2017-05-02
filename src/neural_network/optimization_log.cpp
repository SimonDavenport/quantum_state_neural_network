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

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "optimization_log.hpp"

//!
//! Record a loss record
//!
void OptimizationLog::Record(
    const dvec& parameters,     //!<    Parameters for a group of first trials,
                                //!<    Assumed to be in ascending order
    const double optParameter)  //!<    Optimial refined parameter
{
    this->minLog.push_back(parameters[0]);
    this->maxLog.push_back(parameters[parameters.size()-1]);
    {
        double sumParam = 0;
        for(auto& it : parameters)
        {
            sumParam += it;
        }
        this->meanLog.push_back(sumParam/parameters.size());
    }
    this->optLog.push_back(optParameter);
}

//!
//! Plot the currently stored loss records against a given variable
//!
void OptimizationLog::Plot(
    const bool takeLog,                 //!<    Optionally plot the log of the parameters
    const std::string figName,          //!<    Name of the figure file
    const std::string label,            //!<    Label for the dependent variable
    const std::string paramName,        //!<    Name of the parameter logged
    const std::vector<unsigned int >& dependent)//!<    Values fo the dependent variable
{
    //  Output the log data to a temporary file
    std::ofstream f_out;
    std::string tempFileName = "./log_plot_data.tmp";
    f_out.open(tempFileName);
    if(!f_out.is_open())
    {
        std::cerr << "ERROR: CANNOT CREATE FILE " << tempFileName <<std::endl;
    }
    else
    {
        f_out << dependent.size() << "\n";
        for(auto& it : dependent)
        {
            f_out << it << "\n";
        }
        for(auto& it : minLog)
        {
            f_out << it << "\n";
        }
        for(auto& it : maxLog)
        {
            f_out << it << "\n";
        }
        for(auto& it : meanLog)
        {
            f_out << it << "\n";
        }
        for(auto& it : optLog)
        {
            f_out << it << "\n";
        }
        if(f_out.is_open())
        {
            f_out.close();
        }
        std::string logString = "(";
        if(takeLog)
        {
            logString = "np.log(";
        }
        //  Generate the python script that will perform the plotting
        std::stringstream pythonScript;
        pythonScript.str();
        pythonScript << "#! //usr//bin//env python" << _PYTHON_VERSION_ << "\n"
        "import matplotlib                              \n"
        "import numpy as np                             \n"
        "matplotlib.use('Agg')                        \n\n"
        "import matplotlib.pyplot as plt              \n\n"
        "log_data = np.genfromtxt(\"" << tempFileName << "\")\n"
        "log_size = int(log_data[0])                     \n"
        "x = log_data[1:1+log_size]                     \n"
        "minLog = log_data[1+log_size:1+2*log_size]     \n"
        "maxLog = log_data[1+2*log_size:1+3*log_size]   \n"
        "meanLog = log_data[1+3*log_size:1+4*log_size]  \n"
        "optLog = log_data[1+4*log_size:]               \n"
        "plt.plot(x, " << logString << "minLog), label='min')\n"
        "plt.plot(x, " << logString << "maxLog), label='max')\n"
        "plt.plot(x, " << logString << "meanLog), label='mean')\n"
        "plt.plot(x, " << logString << "optLog), label='opt')\n"
        "plt.legend(loc='best')                         \n"
        "plt.title(\"" << paramName << " vs " << label << "\")\n"
        "plt.savefig(\"" << figName << ".pdf\", bbox_inches=\'tight\') \n"
        "plt.close() \n";
        //  Execute the script
        utilities::Script myScript;
        myScript.SetScript(pythonScript.str());
        myScript.Execute();
        utilities::cout.MainOutput() << "\n\tGenerated plot of optimization log, " 
                                     << "see " << figName << ".pdf\n";
        std::string command = "rm ./log_plot_data.tmp";
        int sysReturn = system(command.c_str()); 
        if(0!=sysReturn)
        {
            std::cerr << sysReturn << std::endl;
        }
    }
}
