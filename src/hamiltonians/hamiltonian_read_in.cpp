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

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "hamiltonian_read_in.hpp"

//!
//! Construct the file name of the model specified by class data
//!
void HamiltonianReadIn::BuildFileName()
{
    std::stringstream ss;
    ss.str("");
    ss << this->inPath;
    if("CD" == this->termType || "DC" == this->termType)
    {
        ss << "quadratic_coefficient_table_";
    }
    else
    {
        ss << "quartic_coefficient_table_";
    }
    if("fqhe_sphere" == modelType)
    {
        ss << "L_" << this->L;
    }
    else
    {
        ss << "kx_" << this->kx << "_ky_" << this->ky;
    }
    ss << ".dat";
    fileName = ss.str();
}

//!
//! Initialize the site label and coefficients containers
//!
void HamiltonianReadIn::InitSiteLabels()
{
    if("DC" == this->termType)
    {
        if("fqhe_sphere" == this->modelType)
        {
            this->siteLabels = new SiteLabels2(this->L);
        }
        else if("ofl" == this->modelType)
        {
            this->siteLabels = new SiteLabels2(this->kx * this->ky);
        }
    }
    else if("DDCC" == this->termType || "CDDC" == this->termType)
    {
        if("fqhe_sphere" == this->modelType)
        {
            this->siteLabels = new SiteLabels4(this->L);
        }
        else if("ofl" == this-> modelType)
        {
            this->siteLabels = new SiteLabels4(this->kx * this->ky);
        }
    }
    else
    {
        std::cerr << "ERROR: Unexpected term type: " << this->termType << std::endl;
    }
    this->siteLabels->Init();
    this->coefficients.resize(this->siteLabels->Count());
}

//!
//! Clear site labels container
//!
void HamiltonianReadIn::ClearSiteLabels()
{
    if(0 != siteLabels)
    {
        delete this->siteLabels;
    }
}

//!
//! Read a single coefficient from a file stream
//!
double HamiltonianReadIn::GetCoefficient(
    std::ifstream& f_in) const
{
    double reCoefficient=0;
    double imCoefficient=0;
    f_in >> reCoefficient;
    if(this->usingImPart)
    {
        f_in >> imCoefficient;
    }
    if("RE" == this->termReIm)
    {
        return reCoefficient;
    }
    else
    {
        return imCoefficient;
    }
}

//!
//! Check if coefficient can be removed from Hamiltonian
//! due to fermion commutation rules. Return true if
//! the coefficient can be set to zero.
//!
bool HamiltonianReadIn::CheckFermionRules(
    std::vector<unsigned int>& kLabels, 
    std::string termType) const
{
    if("DDCC" == termType || "CCDD" == termType)
    {
        return ((kLabels[0] == kLabels[1]) || (kLabels[2] == kLabels[3]));
    }
    if("CDDC" == termType || "DCCD" == termType)
    {
        return ((kLabels[0] == kLabels[3]) || (kLabels[1] == kLabels[2]));
    }
    if("CDCD" == termType || "DCDC" == termType)
    {
        return ((kLabels[0] == kLabels[2]) || (kLabels[1] == kLabels[3]));
    }
    return false;
}

//!
//! Set sign to take into account that the label sorting can 
//! involve commuting fermion operators
//!
double HamiltonianReadIn::CoefficientSign(
    const std::vector<unsigned int>& kLabels,
    const std::vector<unsigned int>& kLabelsSorted) const
{
    double sign = 1.0;
    if("DDCC" == this->termType)
    {
        if(kLabels[0] == kLabelsSorted[1])
        {
            sign *= -1.0;
        }
        if(kLabels[2] == kLabelsSorted[3])
        {
            sign *= -1.0;
        }
    }
    else if("CDDC" == this->termType)
    {
        double sign = 1.0;
        if(kLabels[0] == kLabelsSorted[2])
        {
            sign *= -1.0;
        }
        if(kLabels[2] == kLabelsSorted[3])
        {
            sign *= -1.0;
        }
    }
    else if("CDCD" == this->termType)
    {
        std::cerr << "ERROR: CDCD sign not implemented" << std::endl;
    }
    else
    {
        std::cerr << "ERROR:Unknown term type" << this->termType << std::endl;
    }
    return sign;
}

//!
//! Class initialization
//!
void HamiltonianReadIn::Init()
{
    if("fqhe_sphere" == this->modelType)
    {
        usingImPart = false;
        termReIm = "RE";
    }
    else if("ofl" == this->modelType)
    {
        usingImPart = true;
    }
    utilities::cout.MainOutput() << "\n\tGetting Hamiltonian terms of the form " 
                                 << termType << std::endl;
    this->BuildFileName();
    utilities::cout.MainOutput() << "\tSearching in file " << fileName << "..." << std::endl;
    this->InitSiteLabels();
}

//!
//! Default ctor
//!
HamiltonianReadIn::HamiltonianReadIn()
    :
    termType("CDDC"),
    usingImPart(false),
    termReIm("RE"),
    modelType("ofl"), 
    L(6),
    kx(4),
    ky(6),
    inPath("./"),
    fileName(""),
    siteLabels(0)
{
    this->Init();
};

//!
//! Constructor from command line arguments
//!
HamiltonianReadIn::HamiltonianReadIn(
    boost::program_options::variables_map* optionList,
                                        //!<    Parsed command line argument list
    utilities::MpiWrapper& mpi)         //!<    Address of the mpi wrapper class
    :
    fileName(""),
    siteLabels(0)
{
    GetOption(optionList, this->inPath, "in-path", _LINE_, mpi);
    GetOption(optionList, this->termType, "term-type", _LINE_, mpi);
    GetOption(optionList, this->modelType, "model-type", _LINE_, mpi);
    GetOption(optionList, this->L, "L", _LINE_, mpi);
    GetOption(optionList, this->kx, "kx", _LINE_, mpi);
    GetOption(optionList, this->ky, "ky", _LINE_, mpi);
    this->Init();
}

//!
//! Default dtor
//!
HamiltonianReadIn::~HamiltonianReadIn()
{
    this->ClearSiteLabels();
};

//!
//! Read in non-zero terms of the desired type from a file, 
//! and pad with zeros for other relevant configurations
//!
void HamiltonianReadIn::ImportTerms(
    utilities::MpiWrapper& mpi) //!<    Address of the mpi wrapper class
{
    std::ifstream f_in;
    f_in.open(this->fileName);
    if(!f_in.is_open())
    {
        std::cerr << "\tERROR: CANNOT OPEN FILE " << fileName << std::endl;
        mpi.m_exitFlag = true;
    }
    else
    {
        std::string fileHeader;
        getline(f_in, fileHeader);
        unsigned int nbrTerms;
        unsigned int nbrLabels;
        if(fileHeader == "# File contains term table data in hash table format")
        {
            f_in >> nbrTerms >> nbrLabels;
            //  Skip momentum conserving label combinations data stored in file
            for(unsigned int k=0; k<nbrTerms+1; ++k)
            {
                std::string temp;
                getline(f_in, temp);
            }
            f_in >> nbrTerms >> nbrLabels;
        }
        else if(fileHeader == "# File contains term table data in array format")
        {
            unsigned int highestState;
            f_in >> nbrLabels >> highestState;
            nbrTerms = std::pow(highestState, (int)nbrLabels-1);
        }
        utilities::cout.MainOutput() << "\t...Found "<< nbrTerms << " Hamiltonian terms" 
                                     << " with " << nbrLabels << " labels." << std::endl;
        //  Read in terms of the seleted type only
        std::vector<unsigned int> kLabels(nbrLabels);
        std::vector<unsigned int> kLabelsSorted(nbrLabels);
        std::string storedType = "";
        std::vector<char> charTermType(nbrLabels);
        if(2 == nbrLabels)
        {
            storedType = "DC";
        }
        if(4 == nbrLabels)
        {
            storedType = "DDCC";
        }
        for(unsigned int label=0; label<nbrLabels; ++label)
        {
            charTermType[label] = storedType[label];
        }
        unsigned int termCtr = 0;
        for(unsigned int termIndex=0; termIndex<nbrTerms; ++termIndex)
        {
            std::vector<char> termTypeSorted = charTermType;
            for(auto& it_k : kLabels)
            {
                f_in >> it_k;
            }
            kLabelsSorted = kLabels;
            utilities::QuickSort<unsigned int, char, _ASCENDING_ORDER_>(
                kLabelsSorted.data(), termTypeSorted.data(), kLabelsSorted.size());
            {
                double coefficient = this->GetCoefficient(f_in);
                std::string termType = std::string(termTypeSorted.begin(), termTypeSorted.end());
                if(this->termType ==  termType && !this->CheckFermionRules(kLabelsSorted, termType))
                {
                    if(0.0 != coefficient)
                    {
                        this->coefficients[this->siteLabels->GetIndex(kLabelsSorted)]
                            = this->CoefficientSign(kLabels, kLabelsSorted) * coefficient;
                        ++termCtr;
                    }
                }
            }
        }
        utilities::cout.MainOutput() << "\tImported " << termCtr << " non-zero terms" << std::endl;
        if(0 == termCtr)
        {
            utilities::cout.MainOutput() << "\tExiting due to lack of non-zero terms" << std::endl;
            mpi.m_exitFlag = true;
        }
    }
    if(f_in.is_open())
    {
        f_in.close();
    }
}

//!
//! Return a copy of the array containing all term coefficients
//! stored
//!
void HamiltonianReadIn::GetCoefficients(
    dvec& coefficients)
{
    coefficients = this->coefficients;
}

//!
//! Convert site labels into a set of 
//! binary features
//! 
void HamiltonianReadIn::BuildFeatures(
    utilities::matrix<double>& features)
{
    if(0 != this->siteLabels)
    {
        unsigned int nbrFeatures = this->siteLabels->GetNbrFeatures();
        features.resize(this->siteLabels->Count(), nbrFeatures);
        dvec rowBuffer(nbrFeatures);
        for(unsigned int index=0; index<this->siteLabels->Count(); ++index)
        {
            this->siteLabels->GenerateFeatures(index, rowBuffer);
            features.SetRow(index, rowBuffer);
        }
    }
}
