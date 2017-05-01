////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     Contains a container to store site labels for a collection of 
//!     Hamiltonian terms with ordered labels e.g. sum_{k1>k0} c^+_k0 c_k1.
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

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "site_labels.hpp"

//!
//! Return the number of labels
//!
unsigned int SiteLabels::GetNbrLabels() const
{
    return this->nbrLabels;
}

//!
//! Return the highest state
//!
unsigned int SiteLabels::GetHighestState() const
{
    return this->highestState;
}

//!
//! Get the number of possible binary features that 
//! generated from the site labels
//!
unsigned int SiteLabels::GetNbrFeatures() const
{
    return (1+this->GetNbrLabels()) * (1+this->GetHighestState());
}

//!
//! Print the contents of the site label container
//!
void SiteLabels::Print() const
{
    siteLabels.Print();
}

//!
//! Unique combined key for labels
//!
unsigned int SiteLabels4::LabelKey(
    const unsigned int k0,
    const unsigned int k1,
    const unsigned int k2,
    const unsigned int k3) const
{
    return ((k0*this->highestState+k1)*this->highestState+k2)*this->highestState+k3;
}

//!
//! Recursively count site labels
//!
unsigned int SiteLabels4::Count() const
{
    unsigned int count=0;
    for(unsigned int k0=0; k0<=this->highestState; ++k0)
    {
        for(unsigned int k1=k0+1; k1<=this->highestState; ++k1)
        {
            for(unsigned int k2=k1+1; k2<=this->highestState; ++k2)
            {
                for(unsigned int k3=k2+1; k3<=this->highestState; ++k3)
                {
                    ++count;
                }
            }
        }
    }
    return count;
}

//!
//! Recursively assign site labels
//!
void SiteLabels4::Init()
{
    siteLabels.resize(this->Count(), this->nbrLabels);
    siteLabelIndexes.clear();
    unsigned int index=0;
    for(unsigned int k0=0; k0<=this->highestState; ++k0)
    {
        for(unsigned int k1=k0+1; k1<=this->highestState; ++k1)
        {
            for(unsigned int k2=k1+1; k2<=this->highestState; ++k2)
            {
                for(unsigned int k3=k2+1; k3<=this->highestState; ++k3, ++index)
                {
                    this->siteLabels(index, 0) = k0;
                    this->siteLabels(index, 1) = k1;
                    this->siteLabels(index, 2) = k2;
                    this->siteLabels(index, 3) = k3;
                    this->siteLabelIndexes[this->LabelKey(k0, k1, k2, k3)] = index;
                }
            }
        }
    }
}

//!
//! Get index associated with a given set of keys
//!
unsigned int SiteLabels4::GetIndex(
    const unsigned int k0,
    const unsigned int k1,
    const unsigned int k2,
    const unsigned int k3) const
{
    return this->siteLabelIndexes.at(this->LabelKey(k0, k1, k2, k3));
}

//!
//! Get index associated with a given set of keys
//! as a vector
//!
unsigned int SiteLabels4::GetIndex(
    const std::vector<unsigned int>& kLabels) const
{
    return this->siteLabelIndexes.at(this->LabelKey(kLabels[0], kLabels[1],
                                                    kLabels[2], kLabels[3]));
}

//!
//! Generate binary features for a given set of site
//! occupation labels
//!
void SiteLabels4::GenerateFeatures(
    const unsigned int index,
    std::vector<double>& features) const
{
    if(features.size() != this->GetNbrFeatures())
    {
        features.resize(this->GetNbrFeatures());
    }
    unsigned int k0 = siteLabels(index, 0);
    unsigned int k1 = siteLabels(index, 1);
    unsigned int k2 = siteLabels(index, 2);
    unsigned int k3 = siteLabels(index, 3);
    unsigned int featuresIndex = 0;
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k<k0) ? 1 : 0;
    }
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k>k0 && k<k1) ? 1 : 0;
    }
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k>k1 && k<k2) ? 1 : 0;
    }
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k>k2 && k<k3) ? 1 : 0;
    }
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k>k3) ? 1 : 0;
    }
}

//!
//! Unique combined key for labels
//!
unsigned int SiteLabels2::LabelKey(
    const unsigned int k0,
    const unsigned int k1) const
{
    return k0*this->highestState+k1;
}

//!
//! Recursively count site labels
//!
unsigned int SiteLabels2::Count() const
{
    unsigned int count=0;
    for(unsigned int k0=0; k0<=this->highestState; ++k0)
    {
        for(unsigned int k1=k0+1; k1<=this->highestState; ++k1)
        {
            ++count;
        }
    }
    return count;
}

//!
//! Recursively assign site labels
//!
void SiteLabels2::Init()
{
    siteLabels.resize(this->Count(), this->nbrLabels);
    siteLabelIndexes.clear();
    unsigned int index=0;
    for(unsigned int k0=0; k0<=this->highestState; ++k0)
    {
        for(unsigned int k1=k0+1; k1<=this->highestState; ++k1, ++index)
        {
            this->siteLabels(index, 0) = k0;
            this->siteLabels(index, 1) = k1;
            this->siteLabelIndexes[this->LabelKey(k0, k1)] = index;
        }
    }
}

//!
//! Get index associated with a given set of keys
//!
unsigned int SiteLabels2::GetIndex(
    const unsigned int k0,
    const unsigned int k1) const
{
    return this->siteLabelIndexes.at(this->LabelKey(k0, k1));
}

//!
//! Get index associated with a given set of keys
//! as a vector
//!
unsigned int SiteLabels2::GetIndex(
    const std::vector<unsigned int>& kLabels) const
{
    return this->siteLabelIndexes.at(this->LabelKey(kLabels[0], kLabels[1]));
}

//!
//! Generate binary features for a given set of site
//! occupation labels
//!
void SiteLabels2::GenerateFeatures(
    const unsigned int index,
    std::vector<double>& features) const
{
    if(features.size() != this->GetNbrFeatures())
    {
        features.resize(this->GetNbrFeatures());
    }
    unsigned int k0 = siteLabels(index, 0);
    unsigned int k1 = siteLabels(index, 1);
    unsigned int featuresIndex = 0;
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k<k0) ? 1 : 0;
    }
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k>k0 && k<k1) ? 1 : 0;
    }
    for(unsigned int k=0; k<=this->highestState; ++k, ++featuresIndex)
    {
        features[featuresIndex] = (k>k1) ? 1 : 0;
    }
}
