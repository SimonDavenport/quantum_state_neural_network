////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     Contains a container to store site labels for a collection of 
//!     Hamiltonian terms with ordered labels e.g. 
//!     sum_{k1>k0} X(k0, k1) c^+_k0 c_k1.
//!     Such terms have simple representations in terms of matrix 
//!     product operators. X(k0, k1) is a coefficient which can be 
//!     expressed in terms of a product of site matrices raised to 
//!     the power of a binary representing whether that site is
//!     "empty". Those binaries can be thought of as "features".
//!
//!     The "features" for a given set of site labels correspond 
//!     to actions on empty sites between those sites where 
//!     operators act. 
//!     Example: acting on base features 4, 5, 7 ,8 for L=10, 
//!     corresponds to no action on sites 0, 1, 2, 3, 6, 9, 10.
//!     The non action on each site is a binary feature. Features are 
//!     further distinguished by where they occur compared to where
//!     the operators act. The above corresponds to the features:
//!     1 1 1 1 0 0 0 0 0 0 0,
//!     0 0 0 0 0 0 0 0 0 0 0, 
//!     0 0 0 0 0 0 1 0 0 0 0, 
//!     0 0 0 0 0 0 0 0 0 0 0,
//!     0 0 0 0 0 0 0 0 1 1 1
//!     In general there are (nbr_labels+1) * (highest_state+1)
//!     possible features in this scheme.
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
  
#ifndef _SITE_LABELS_HPP_INCLUDED_
#define _SITE_LABELS_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include <unordered_map>
#include "../utilities/general/dvec_def.hpp"
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/algorithms/quick_sort.hpp"
#include "../utilities/wrappers/io_wrapper.hpp"

//!
//! Base class for site labels container
//!
class SiteLabels
{
    protected:
    unsigned int nbrLabels;
    unsigned int highestState;
    utilities::matrix<double> siteLabels;
    std::unordered_map<unsigned int, unsigned int> siteLabelIndexes;
    
    virtual unsigned int LabelKey(
        const unsigned int k0,
        const unsigned int k1,
        const unsigned int k2,
        const unsigned int k3) const {return 0;};
    virtual unsigned int LabelKey(
        const unsigned int k0,
        const unsigned int k1) const {return 0;};

    public:
    SiteLabels()
        :
        nbrLabels(0),
        highestState(0)
    {};
    SiteLabels(
        const unsigned int nbrLabels,
        const unsigned int highestState)
        :
        nbrLabels(nbrLabels),
        highestState(highestState)
    {};
    virtual ~SiteLabels(){};
    unsigned int GetNbrLabels() const;
    unsigned int GetHighestState() const;
    unsigned int GetNbrFeatures() const;
    void Print() const;
    virtual unsigned int Count() const=0;
    virtual void Init()=0;
    virtual unsigned int GetIndex(
        const unsigned int k0,
        const unsigned int k1) const {return 0;};
    virtual unsigned int GetIndex(
        const unsigned int k0,
        const unsigned int k1,
        const unsigned int k2,
        const unsigned int k3) const {return 0;};
    virtual unsigned int GetIndex(
        const std::vector<unsigned int>& kLabels) const=0;
    virtual void GenerateFeatures(const unsigned int index,
                                  std::vector<double>& features) const=0;
};

//!
//! Container for an array of site labels 
//! k0, k1, k2, k3 stored in ascending order
//!
class SiteLabels4 : public SiteLabels
{
    private:
    unsigned int LabelKey(
        const unsigned int k0,
        const unsigned int k1,
        const unsigned int k2,
        const unsigned int k3) const;
    
    public:
    //!
    //! Default ctor
    //!
    SiteLabels4()
        :
        SiteLabels(4, 0)
    {}
    //!
    //! Ctor given highest state
    //!
    SiteLabels4(
        const unsigned int highestState)
        :
        SiteLabels(4, highestState)
    {}
    unsigned int Count() const;
    void Init();
    unsigned int GetIndex(
        const unsigned int k0, const unsigned int k1,
        const unsigned int k2, const unsigned int k3) const;
    unsigned int GetIndex(
        const std::vector<unsigned int>& kLabels) const;
    void GenerateFeatures(const unsigned int index, 
                          std::vector<double>& features) const;
};

//!
//! Container for an array of site labels 
//! k0, k1 stored in ascending order
//!
class SiteLabels2 : public SiteLabels
{
    private:
    //!
    //! Unique combined key for labels
    //!
    unsigned int LabelKey(
        const unsigned int k0,
        const unsigned int k1) const;
    
    public:
    //!
    //! Default ctor
    //!
    SiteLabels2()
        :
        SiteLabels(2, 0)
    {}
    //!
    //! Ctor given highest state
    //!
    SiteLabels2(
        const unsigned int highestState)
        :
        SiteLabels(2, highestState)
    {}
    unsigned int Count() const;
    void Init();
    unsigned int GetIndex(
        const unsigned int k0, const unsigned int k1) const;
    unsigned int GetIndex(
        const std::vector<unsigned int>& kLabels) const;
    void GenerateFeatures(const unsigned int index,
                          std::vector<double>& features) const;
};
#endif
