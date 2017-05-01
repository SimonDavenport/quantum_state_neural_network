////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		Run tests for the implementation of the SiteLables containers,
//!     allowing read in of site labels and generation of derivaed features
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
#include "../features/site_labels.hpp"
#include "../utilities/linear_algebra/dense_matrix.hpp"

int main(int argc, char *argv[])
{
    std::cout << "TEST SITE LABELS AND FEATURE DERIVATION IMPLEMENTATION" << std::endl;
    {
        std::cout << "2 labels" << std::endl;
        unsigned int highestState = 6;
        SiteLabels2 siteLabels(highestState);
        siteLabels.Init();
        std::cout << "k0\tk1\tindex" << std::endl;
        std::vector<unsigned int> k(2);
        for(unsigned int k0=0; k0<=highestState; ++k0)
        {
            for(unsigned int k1=k0+1; k1<=highestState; ++k1)
            {
                k[0] = k0;
                k[1] = k1;
                std::cout << k0 << "\t" << k1 << "\t" << siteLabels.GetIndex(k) << std::endl;
            }
        }
        std::cout << "Matrix form:" << std::endl;
        siteLabels.Print();
        unsigned int nbrFeatures = siteLabels.GetNbrFeatures();
        utilities::matrix<double> features(siteLabels.Count(), nbrFeatures);
        dvec rowBuffer(nbrFeatures);
        for(unsigned int index=0; index<siteLabels.Count(); ++index)
        {
            siteLabels.GenerateFeatures(index, rowBuffer);
            features.SetRow(index, rowBuffer);
        }
        std::cout << "Binary features:" << std::endl;
        features.Print();
    }
    {
        std::cout << "4 labels" << std::endl;
        unsigned int highestState = 6;
        SiteLabels4 siteLabels(highestState);
        siteLabels.Init();
        std::cout << "k0\tk1\tk2\tk3\tindex" << std::endl;
        std::vector<unsigned int> k(4);
        for(unsigned int k0=0; k0<=highestState; ++k0)
        {
            for(unsigned int k1=k0+1; k1<=highestState; ++k1)
            {
                for(unsigned int k2=k1+1; k2<=highestState; ++k2)
                {
                    for(unsigned int k3=k2+1; k3<=highestState; ++k3)
                    {
                        k[0] = k0;
                        k[1] = k1;
                        k[2] = k2;
                        k[3] = k3;
                        std::cout << k0 << "\t" << k1 << "\t" << k2 << "\t" << k3
                                  << "\t" << siteLabels.GetIndex(k) << std::endl;
                    }
                }
            }
        }
        std::cout << "Matrix form:" << std::endl;
        siteLabels.Print();
        unsigned int nbrFeatures = siteLabels.GetNbrFeatures();
        utilities::matrix<double> features(siteLabels.Count(), nbrFeatures);
        dvec rowBuffer(nbrFeatures);
        for(unsigned int index=0; index<siteLabels.Count(); ++index)
        {
            siteLabels.GenerateFeatures(index, rowBuffer);
            features.SetRow(index, rowBuffer);
        }
        std::cout << "Binary features:" << std::endl;
        features.Print();
    }
    return EXIT_SUCCESS;
}
