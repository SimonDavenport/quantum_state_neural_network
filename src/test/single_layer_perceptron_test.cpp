////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		Run tests for the implementation of the single layer perceptron 
//!     neutral network.
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
#include "../neural_network/single_layer_perceptron.hpp"
#include "../utilities/general/dvec_def.hpp"
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/wrappers/io_wrapper.hpp"

#if _DEBUG_
#include "../utilities/general/debug.hpp"
#endif

int main(int argc, char *argv[])
{   
    const unsigned int N = 20;
    const unsigned int trainN = 20;
    const unsigned int testN = 0;
    const unsigned int H = 10;
    const unsigned int P = 2;
    //  Get a dataset and partition into train and test subsets
    utilities::matrix<double> features(P, N);
    dvec outputs(N);
    std::ifstream f_features;
    utilities::GenFileStream(f_features, "./test/test_features.dat", io::_TEXT_);
    for(auto& it : features.m_data)
    {
        f_features >> it;
    }
    f_features.close();
    std::ifstream f_outputs;
    utilities::GenFileStream(f_outputs, "./test/test_outputs.dat", io::_TEXT_);
    for(auto& it : outputs)
    {
        f_outputs >> it;
    }
    f_outputs.close();

    utilities::matrix<double> trainFeatures(P, trainN);
    dvec trainOutputs(trainN);
    utilities::ToSubMatrix(trainFeatures, features, 0, 0);
    utilities::ToSubVector(trainOutputs, outputs, 0);
    utilities::matrix<double> testFeatures(P, testN);
    dvec testOutputs(testN);
    utilities::ToSubMatrix(testFeatures, features, 0, trainN);
    utilities::ToSubVector(testOutputs, outputs, trainN);

    //  Train the single layer perceptron
    ann::SingleLayerPerceptron slp(P, H);
    slp.AllocateWork(trainN);
    slp.RandomizeWeights(0.5, 0);
    ann::Train(slp, trainOutputs, trainFeatures);
    dvec networkOutputs(N);
    slp.Evaluate(networkOutputs, features);
    std::cout << "\t" << "Prediction" << " " << "Actual" << std::endl;
    for(auto it_network = networkOutputs.begin(), it_test = outputs.begin();
        it_network < networkOutputs.end(); ++it_network, ++it_test)
    {
        std::cout << "\t" << *it_network << " " << *it_test << std::endl;
    }

    return EXIT_SUCCESS;
}
