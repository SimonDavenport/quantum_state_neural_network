////////////////////////////////////////////////////////////////////////////////
//!                                                                             
//!                        \author Simon C. Davenport
//!                                                                             
//!	 \file	
//!     A wrapper around file io to allow for different file formats
//!
////////////////////////////////////////////////////////////////////////////////

#ifndef _IO_WRAPPER_HPP_INCLUDED_
#define _IO_WRAPPER_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <fstream>
#include "../wrappers/mpi_wrapper.hpp"
#include "../general/template_tools.hpp"

namespace utilities
{
    template<typename F>
    F GenFileStream(const std::string fileName, std::string format, utilities::MpiWrapper& mpi)
    {
        F stream;
        if("binary" == format)
        {
            stream.open(fileName.c_str(), std::ios::binary);
        }
        else if("text" == format)
        {
            if(utilities::is_same<F, std::ifstream>::value)
            {
                stream.open(fileName.c_str(), std::ios::in);
            }
            else if(utilities::is_same<F, std::ofstream>::value)
            {
                stream.open(fileName.c_str(), std::ios::out);
            }
        }
        else
        {
            std::cerr << "Unknown file format " << format << std::endl;
            mpi.m_exitFlag=true;
        }
        if(!mpi.m_exitFlag && !stream.is_open())
        {
            std::cerr << "Could not open file " << fileName << std::endl;
            mpi.m_exitFlag=true;
        }
        return F;
    }
}   //  End namespace utilities
#endif 
