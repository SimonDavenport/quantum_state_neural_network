////////////////////////////////////////////////////////////////////////////////
//!                                                                             
//!                        \author Simon C. Davenport
//!                                                                             
//!	 \file
//!     The file contains an interface class that allows for selected
//!     read in of Hamiltonian term data generated from the 
//!     hamiltonian_diagonalization program.
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

#ifndef _HAMILTONIAN_READ_IN_HPP_INCLUDED_
#define _HAMILTONIAN_READ_IN_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../features/site_labels.hpp"
#include "../utilities/general/dvec_def.hpp"
#include "../utilities/general/cout_tools.hpp"
#include "../utilities/wrappers/mpi_wrapper.hpp"
#include "../utilities/wrappers/program_options_wrapper.hpp"

namespace myOptions
{
    namespace po = boost::program_options;
    inline po::options_description GetHamiltonianReadInOptions()
    {
        po::options_description hamiltonianOpt("Hamiltonian read in options");
        hamiltonianOpt.add_options()
        ("term-type", po::value<std::string>()->default_value("DDCC"),
         "Specify the type of Hamiltonian to examine. Format is D representing a c^+ and C representing a c operator. \nThe labels of the oeprators will be constructed increasing from left to right.")
        ("model-type", po::value<std::string>()->default_value("fqhe_sphere"), 
         "Name of the model being considered. Options are: \n fqhe_sphere, ofl.")
        ("L", po::value<unsigned int>()->default_value(6), 
         "Highest orbital for fqhe_sphere.")
        ("kx", po::value<unsigned int>()->default_value(2), 
         "x dimension of lattice for ofl.")
        ("ky", po::value<unsigned int>()->default_value(2), 
         "y dimension of lattice for ofl.");
        return hamiltonianOpt;
    };
}   //  End namespace myOptions

//!
//! Container for extraction of Hamiltonian terms.
//! We need to be able to select what terms to fit
//! for a given neural network, as typically only
//! certian sets of terms can be treated in one network.
//!
class HamiltonianReadIn
{
    private:
    std::string termType;
    bool usingImPart;
    std::string termReIm;
    std::string modelType;
    unsigned int L;
    unsigned int kx;
    unsigned int ky;
    std::string inPath;
    std::string fileName;
    SiteLabels* siteLabels;
    dvec coefficients;
    
    void BuildFileName();
    void InitSiteLabels();
    void ClearSiteLabels();
    double GetCoefficient(std::ifstream& f_in) const;
    bool CheckFermionRules(
        std::vector<unsigned int>& kLabels, 
        std::string termType) const;
    double CoefficientSign(
        const std::vector<unsigned int>& kLabels,
        const std::vector<unsigned int>& kLabelsSorted) const;
    void Init();
    public:
    HamiltonianReadIn();
    HamiltonianReadIn(
        boost::program_options::variables_map* optionList,    
        utilities::MpiWrapper& mpi);
    ~HamiltonianReadIn();
    void ImportTerms(utilities::MpiWrapper& mpi);
    void GetCoefficients(dvec& coefficients);
    void BuildFeatures(utilities::matrix<double>& features);
};
#endif
