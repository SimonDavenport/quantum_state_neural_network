////////////////////////////////////////////////////////////////////////////////
//!                                                                             
//!                        \author Simon C. Davenport
//!                                                                                                        
//!	 \file
//!     This file contains debugging tools      
//!                                                        
//!                    Copyright (C) Simon C Davenport
//!                                                                             
//!      This program is free software: you can redistribute it and/or modify
//!      it under the terms of the GNU General Public License as published by
//!      the Free Software Foundation, either version 3 of the License,
//!      or (at your option) any later version.
//!                                                                             
//!      This program is distributed in the hope that it will be useful, but
//!      WITHOUT ANY WARRANTY; without even the implied warranty of
//!      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//!      General Public License for more details.
//!                                                                             
//!      You should have received a copy of the GNU General Public License
//!      along with this program. If not, see <http://www.gnu.org/licenses/>.
//!                                                                             
//////////////////////////////////////////////////////////////////////////////// 

#ifndef _DEBUG_TOOLS_HPP_INCLUDED_
#define _DEBUG_TOOLS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////		
#include <iostream>
#include <string>
#include <bitset>
#include <vector>

inline void BREAK(const int val)
{
    std::cout<<"\n\tHERE "<<val<<std::endl;
    return;
}

inline void PAUSE()
{
    std::cout<<"______________________________________________"<<std::endl;
    getchar();
    return;
}

template <typename T>
void PRINT(const std::string name, const T val)
{
    std::cout << "\n\t" << name << " = " << val << std::endl;
    return;
}

template <typename T>
void PRINT(const std::string name, T* array, const int dim)
{
    std::cout<<std::endl;
    for(int i=0; i<dim; ++i)
    {
        std::cout << "\t" << name << "[" << i << "] = " << array[i] << std::endl;
    }
    std::cout << std::endl;
    return;
}

template<typename T>
void PRINTVEC(const std::string name, std::vector<T>& vec)
{
    int dim = vec.size();
    PRINT<T>(name, vec.data(), dim);
}
#endif
