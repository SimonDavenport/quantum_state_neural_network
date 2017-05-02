////////////////////////////////////////////////////////////////////////////////
//!                                                                             
//!                        \author Simon C. Davenport
//!                                                                             
//!	 \file
//!     This file declares the general program options to be used by any main
//!     program.
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

#ifndef _GENERAL_OPTIONS_HPP_INCLUDED_
#define _GENERAL_OPTIONS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <boost/program_options.hpp>

namespace myOptions
{
    namespace po = boost::program_options;
    inline po::options_description GetGeneralOptions()
    {
        po::options_description generalOpt("General Options");
        generalOpt.add_options()
        ("help,h",
         "Display this message\n")
        ("verbose,v", po::value<int>()->default_value(1),
         "Set a value for the verbosity level:\n\t 0 output off (after command line parsed) \n\t 1 print brief information \n\t 2 print more detailed information \n\t 3 also print loading bars and timings \n\t 4 also print debugging messages.")
        ("in-path", po::value<std::string>()->default_value("./"),
         "Specify the name of the file system path where program input data are stored\n")
        ("out-path", po::value<std::string>()->default_value("./"),
         "Specify the name of the file system path where program output data are stored\n");
        return generalOpt;
    };
}   //  End namespace myOptions
#endif
