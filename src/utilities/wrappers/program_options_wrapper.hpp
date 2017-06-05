////////////////////////////////////////////////////////////////////////////////
//!                                                                             
//!                        \author Simon C. Davenport
//!                                                                             
//!	 \file
//!     This file declares a thin wrapper to allow reading of program options
//!     such that option parsing exceptions are automatically handeled.
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

#ifndef _PROGRAM_OPTIONS_WRAPPER_HPP_INCLUDED_
#define _PROGRAM_OPTIONS_WRAPPER_HPP_INCLUDED_

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define _LINE_ __FILE__ ":" TOSTRING(__LINE__)

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <boost/program_options.hpp>
#include "../wrappers/mpi_wrapper.hpp"

namespace utilities
{
    //!
    //! Wrapper around program options parser
    //!
    template <typename T>
    void GetOption(
        boost::program_options::variables_map* variables_map,   //!< Pointer to map of command line variables
        T& variable,                //!<    Address where variable will be stored
        const std::string name,     //!<    Name of variable to be used
        const std::string location) //!<    Location of the function call
    {
        try
	    {
	        variable = (*variables_map)[name].as<T>();
	    }
	    catch(boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_any_cast> >& e)
	    {
	        std::cerr << "\n\tERROR Option " << name << " queried from variables_map incorrectly in ";
	        std::cerr << location  << std::endl;
	        exit(EXIT_FAILURE);
	    }
    }

    //!
    //! Wrapper around program options parser (including action in parallel implementaion)
    //!
    template <typename T>
    void GetOption(
        boost::program_options::variables_map* variables_map,   //!< Pointer to map of command line variables
        T& variable,                //!<    Address where variable will be stored
        const std::string name,     //!<    Name of variable to be used
        const std::string location, //!<    Location of the function call
        utilities::MpiWrapper& mpi) //!<    Address of mpi wrapper
    {
        try
	    {
	        variable = (*variables_map)[name].as<T>();
	    }
	    catch(boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_any_cast> >& e)
	    {
	        std::cerr << "\n\tERROR Option " << name << " queried from variables_map incorrectly in ";
	        std::cerr << location  << std::endl;
	        mpi.m_exitFlag = true;
	    }
    }
    
}   //  End namespace utilities
#endif
