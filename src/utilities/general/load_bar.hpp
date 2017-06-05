////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This file contains a function to display a loading bar on the command
//!     line
//!
//!                    Copyright (C) Simon C Davenport
//!
//!		This program is free software: you can redistribute it and/or modify
//!		it under the terms of the GNU General Public License as published by
//!		the Free Software Foundation, either version 3 of the License,
//!		or (at your option) any later version.
//!
//!		This program is distributed in the hope that it will be useful, but
//!		WITHOUT ANY WARRANTY; without even the implied warranty of 
//!		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
//!		General Public License for more details.
//!
//!		You should have received a copy of the GNU General Public License
//!		along with this program. If not, see <http://www.gnu.org/licenses/>.
//! 
////////////////////////////////////////////////////////////////////////////////  

#ifndef _LOAD_BAR_HPP_INCLUDED_
#define _LOAD_BAR_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////		 
#include "cout_tools.hpp"

namespace utilities
{

//////////////////////////////////////////////////////////////////////////////////
//! \brief A class to display a progress bar 
//////////////////////////////////////////////////////////////////////////////////

class LoadBar
{
    private:
    long int m_max;                     //!<    Number corresponding to 100% completion
    long int m_onePercent;              //!<    An integer closest to 1% of the total
    static const int m_barWidth = 50;   //!<    Character width of the display bar
    
    public:
    //!
    //!  Default constructor
    //!
    LoadBar()
    :
        m_max(0),
        m_onePercent(0)
    {}
    //!
    //!  Initialize function
    //!
    void Initialize(
        const long int max) //!<   Number of values in total 
    {
        m_max = max;
        m_onePercent = ceil((double)m_max/100);
    }
    //!
    //! Destructor
    //!
    ~LoadBar()
    {
        //utilities::cout.AdditionalInfo()<<std::endl;
    }
    //!
    //! Display current progress
    //!
    inline void Display(
        const long int x) //!<   Position you've got to
        const
    {
        if(x % m_onePercent != 0 && (m_max-x)>m_onePercent) return;
        double ratio = (double)x/m_max;
        int c = round(ratio * m_barWidth);
        utilities::cout.AdditionalInfo()<<"\t"<<(int)(ratio*100)<<"% [";
        for (int i=0; i<c; ++i)
        {
            utilities::cout.AdditionalInfo()<<"=";
        }
        for (int i=c; i<m_barWidth; ++i)
        {
            utilities::cout.AdditionalInfo()<<" ";
        }
        utilities::cout.AdditionalInfo()<<"]\r";
        fflush(stdout);
        return;
    }
};      //  End LoadBar class
}       //  End namespace utilities
#endif
