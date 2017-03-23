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
#include <iostream>
#include <iomanip>
//#include "../wrappers/mpi_wrapper.hpp"
#include "../general/template_tools.hpp"
#include "../general/dcmplx_type_def.hpp"

namespace io
{
    //////      ENUM DECLARATIONS       ////////////////////////////////////////
    enum fileFormat_t {_TEXT_=0, _BINARY_=1};
    //!<    Define a list of file format types
    enum io_t {_IN_, _OUT_};
    //!<    Define a type for specifying construction of input or output files
}

namespace utilities
{
    //!
    //! Wrapper around file stream opening checks
    //!
    template<typename F>
    void GenFileStream(F& stream, const std::string fileName, io::fileFormat_t format)//, utilities::MpiWrapper& mpi)
    {
        if(io::_BINARY_ == format)
        {
            if(utilities::is_same<F, std::ifstream>::value)
            {
                stream.open(fileName.c_str(), std::ios::in | std::ios::binary);
            }
            else if(utilities::is_same<F, std::ofstream>::value)
            {
                stream.open(fileName.c_str(), std::ios::out | std::ios::binary);
            }
        }
        else if(io::_TEXT_ == format)
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
            //mpi.m_exitFlag=true;
        }
        //if(!mpi.m_exitFlag && !stream.is_open())
        if(!stream.is_open())
        {
            std::cerr << "Could not open file " << fileName << std::endl;
            //mpi.m_exitFlag=true;
        }
    }
    
    //!
    //! Template function to parse different data types. In particular complex
    //! types don't have native parsing so need to be treated as a special case
    //!
    template<typename T>
    std::string ToStream(T variable)
    {
        std::stringstream ss;
        ss.str("");
        if(utilities::is_same<dcmplx, T>::value)
        {
            double re, im;
            re = std::real(variable);
            im = std::imag(variable);
            ss << std::setprecision(15) << re << "\t" << std::setprecision(15) << im;
        }
        else
        {
            ss << std::setprecision(15) << variable;
        }
        return ss.str();
    }
    
    //!
    //! Read variable from a stream
    //!
    inline void FromStream(std::ifstream& stream, double& value)
    {
        stream >> value;
    }
    
    //!
    //! Read variable from a stream
    //!
    inline void FromStream(std::ifstream& stream, short unsigned int& value)
    {
        stream >> value;
    }
    
    //!
    //! Overload of FromStream deals with dcomplx types
    //!
    inline void FromStream(std::ifstream& stream, dcmplx& value)
    {
        double re, im;
        stream >> re >> im;
        value = std::complex<double>(re, im);
    }
    
}   //  End namespace utilities
#endif 
