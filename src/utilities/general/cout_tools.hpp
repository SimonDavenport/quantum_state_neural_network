////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport
//!
//!  \file
//!		Header file for cout tools
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

#ifndef _COUT_TOOLS_HPP_INCLUDED_
#define _COUT_TOOLS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <mpi.h>
#include <iostream>
#include <fstream>

namespace utilities
{
    enum vLevel {_OUTPUT_OFF_=0, _MAIN_OUTPUT_=1, _SECONDARY_OUTPUT_=2, 
                 _ADDITIONAL_INFO_=3, _DEBUGGING_INFO_=4};
    
    ////////////////////////////////////////////////////////////////////////////////
    //! \brief Define a class to contain functions that control the verbosity 
    //! level of cout output. An object of this class type should be declared 
    //! as a global extern, with a single instance declared in the main file.
    ////////////////////////////////////////////////////////////////////////////////

    class Cout
    {
        private:
        vLevel m_verbosityLevel;    //!<    A signed integer to define the verbosity
                                    //!     level: -1 turns off all output, and
                                    //!     positive values specify increasingly
                                    //!     detailed levels of output
        std::ofstream m_nullOut;    //!<    An output stream
                                    //!     to the null device - this can be 
                                    //!     used to redirect unwanted output to 
                                    //!     be discarded
        //!
        //! An output level function to optionally print various levels
        //! of output, or alternatively send output to /dev/null
        //!
        inline std::ostream& Level(
            const vLevel& level)    //!<    Specified output level
        {
            if(m_verbosityLevel >= level && _OUTPUT_OFF_ != level)
            {
                return std::cout;
            }
            else
            {
                return m_nullOut;
            }
        }
        
        public:
        //!
        //! Default constructor
        //!
        Cout()
            :   m_verbosityLevel(_MAIN_OUTPUT_),
                m_nullOut("/dev/null")
                
        {}
        //!
        //!  A function to set the verbosity level
        //!
        inline void SetVerbosity(
            const short int verbosityLevel)     //!<    New verbosity level
        {
            switch(verbosityLevel)
            {
                case 0: 
                    m_verbosityLevel = _OUTPUT_OFF_;
                    break;
                case 1:
                    m_verbosityLevel = _MAIN_OUTPUT_;
                    break;
                case 2:
                    m_verbosityLevel = _SECONDARY_OUTPUT_;
                    break;
                case 3:
                    m_verbosityLevel = _ADDITIONAL_INFO_;
                    break;
                case 4:
                    m_verbosityLevel = _DEBUGGING_INFO_;
                    break;
                default:
                    m_verbosityLevel = _MAIN_OUTPUT_;
            }
        }
        //!
        //!  A function to get the verbosity level
        //!
        inline short int GetVerbosity() const
        {
            return m_verbosityLevel; 
        }
        //!
        //! Public interface for the Level function - main output
        //!
        inline std::ostream& MainOutput()
        {
            return this->Level(_MAIN_OUTPUT_);
        }
        //!
        //! Public interface for the Level function - secondary output
        //!
        inline std::ostream& SecondaryOutput()
        {
            return this->Level(_SECONDARY_OUTPUT_);
        }
        //!
        //! Public interface for the Level function - additional info
        //!
        inline std::ostream& AdditionalInfo()
        {
            return this->Level(_ADDITIONAL_INFO_);
        }
        //!
        //! Public interface for the Level function - debugging info
        //!
        inline std::ostream& DebuggingInfo()
        {
            return this->Level(_DEBUGGING_INFO_);
        }
        //!
        //! MPI sync function to set the same verbosity on all nodes
        //! (can only be called after MPI_Init)
        //!
        inline void MpiSync(
            const int syncNodeId,   //!<    Node ID to sync with
            MPI_Comm comm)          //!<    MPI communicator
        {
            MPI_Bcast(&m_verbosityLevel, 1, MPI_INT, syncNodeId, comm);
        }
        //!
        //! Make a string containing a line of hyphens
        //!
        inline std::string HyphenLine()
        {
            return "---------------------------------------------"
                   "---------------------------------------------";
        }
        //!
        //! Make a string containing a line of equals
        //!
        inline std::string EqualsLine()
        {
            return "============================================="
                   "=============================================";
        }
    };
    extern Cout cout;      //!< Specify an extern declaration meaning that 
                           //!  every file including coutTools.hpp expects
                           //!  an Cout struct to be declared once
}   //  End namespace utilities
#endif
