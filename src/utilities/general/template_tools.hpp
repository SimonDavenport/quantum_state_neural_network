////////////////////////////////////////////////////////////////////////////////
//!                                                                             
//!                        \author Simon C. Davenport
//!                                                                                                  
//!	 \file
//!		This file contains some implementations of template utilities that 
//!     are included in C++ 11 's type_traits library. 
//!     
//!     Also there are some utility functions for "variadic templates".
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

#ifndef _TEMPLATE_TOOLS_HPP_INCLUDED_
#define _TEMPLATE_TOOLS_HPP_INCLUDED_

namespace utilities
{
    //!
    //! A template to check that two types are the same
    //! (as implemented by the function std::is_same in C++11)
    //!
    template<typename T, typename U>
    struct is_same 
    {
        static const bool value = false; 
    };
    
    //!
    //! A template to check that two types are the same
    //! (as implemented by the function std::is_same in C++11)
    //!
    template<typename T>
    struct is_same<T, T>
    { 
       static const bool value = true; 
    };
}   //  End namespace utilities
#endif
