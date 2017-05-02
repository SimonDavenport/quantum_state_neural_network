////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     This program reads in a set of coefficieitns for Hamiltonain terms,
//!     then attempts to fit these in terms of a certian neural network
//!     model. This model has an equivalent representation in terms of
//!     finite state machines, so the method allows for a compact
//!     representation of the Hamiltonian in terms of an approximate
//!     set of finte state machine terms. Such a representation is 
//!     extremely useful for applications to matrix product operator states.
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
#include "../neural_network/train.hpp"
#include "../utilities/general/dvec_def.hpp"
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/optimization/bfgs.hpp"
#include "../utilities/optimization/lbfgs.hpp"
#include "../utilities/algorithms/quick_sort.hpp"
#include "../utilities/general/run_script.hpp"
#include "../features/site_labels.hpp"

//!
//! Container for extration of Hamiltonian terms.
//! We need to be able to select what terms to fit
//! for a given neural network, as typically only
//! certian sets of terms can be treated in one network.
//!
class Hamiltonian
{
    private:
    std::string termType;
    bool usingImPart;
    std::string termReIm;
    std::string modelType;
    unsigned int L;
    unsigned int kx;
    unsigned int ky;
    std::string fileName;
    SiteLabels* siteLabels;
    dvec coefficients;

    //!
    //! Construct the file name of the model specified by class data
    //!
    void BuildFileName()
    {
        std::stringstream ss;
        ss.str("");
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
    void InitSiteLabels()
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
            exit(EXIT_FAILURE);
        }
        this->siteLabels->Init();
        this->coefficients.resize(this->siteLabels->Count());
    }
    
    //!
    //! Clear site labels container
    //!
    void ClearSiteLabels()
    {
        if(0 != siteLabels)
        {
            delete this->siteLabels;
        }
    }

    //!
    //! Read a single coefficient from a file stream
    //!
    double GetCoefficient(
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
    bool CheckFermionRules(
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
    double CoefficientSign(
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
            std::cerr << "CDCD sign not implemented" << std::endl;
            exit(EXIT_FAILURE);
        }
        return sign;
    }

    public:
    //!
    //! Default ctor
    //!
    Hamiltonian()
        :   
        termType("CDDC"),
        usingImPart(false),
        termReIm("RE"),
        modelType("ofl"), 
        L(6),
        kx(4),
        ky(6),
        fileName(),
        siteLabels(0)
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
        std::cout << "Getting Hamiltonian terms of the form " << termType <<std::endl;
        this->BuildFileName();
        this->InitSiteLabels();
    };
    
    ~Hamiltonian()
    {
        this->ClearSiteLabels();
    };
    
    //!
    //! Read in non-zero terms of the desired type from a file, 
    //! and pad with zeros for other relevant configurations
    //!
    void ImportTerms()
    {
        std::ifstream f_in;
        f_in.open(this->fileName);
        if(!f_in.is_open())
        {
            std::cerr << " CANNOT OPEN FILE " << fileName << std::endl;
            exit(EXIT_FAILURE);
        }
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
        std::cout << nbrTerms << " " << nbrLabels << std::endl;
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
        if(f_in.is_open())
        {
            f_in.close();
        }
        std::cout << "Imported " << termCtr << " non-zero terms" << std::endl;
    }
    
    //!
    //! Return a copy of the array containing all term coefficients
    //! stored
    //!
    void GetCoefficients(
        dvec& coefficients)
    {
        coefficients = this->coefficients;
    }
    
    //!
    //! Convert site labels into a set of 
    //! binary features
    //! 
    void BuildFeatures(
        utilities::matrix<double>& features)
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
};

//!
//! Divide features and outputs into training and testing sets
//!
void GetTrainTestDivision(
    utilities::matrix<double>& features,
    utilities::matrix<double>& trainFeatures,
    utilities::matrix<double>& testFeatures,
    dvec& outputs, 
    dvec& trainOutputs,
    dvec& testOutputs)
{
    utilities::ToSubMatrix(trainFeatures, features, 0, 0);
    utilities::ToSubVector(trainOutputs, outputs, 0);
    if(testOutputs.size())
    {
        utilities::ToSubMatrix(testFeatures, features, 0, trainOutputs.size());
        utilities::ToSubVector(testOutputs, outputs, trainOutputs.size());
    }
}

//!
//! Set loss function residual weights to penalize
//! more or less heavily the network outputs that
//! are expected to be exaclty zero
//!
void SetResidualWeights(
    ann::LossFunctionWeights& lfw,
    dvec& outputs)
{
    dvec residualWeights(outputs.size());
    for(unsigned int index=0; index<outputs.size(); ++index)
    {
        if(outputs[index] == 0.0)
        {
            residualWeights[index] = 10.0;
        }
        else
        {
            residualWeights[index] = 1.0;
        }
    }
    lfw.SetResidualWeights(residualWeights);
}

//!
//! Perform a first sweep optimization of the network
//! starting from a random configuration specified by 
//! the seed
//!
void NetworkOptimizationFirstSweep(
    ann::SingleLayerPerceptron& slp,
    dvec& trainOutputs, 
    utilities::matrix<double> trainFeatures,
    unsigned int seed)
{
    utilities::optimize::LBFGS op;
    ann::LossFunctionWeights lfw;
    SetResidualWeights(lfw, trainOutputs);
    //  Perform a few iterations with a heavy L1 penalty
    //  on the alpha weights, then prune small weights
    {
        const unsigned int maxIter0 = 20;
        const double gradTol0 = 1e-2;
        lfw.l1Alpha = 1.0;
        slp.SetLossFunctionWeights(lfw);
        slp.RandomizeWeights(0.5, seed);
        ann::Train(slp, op, trainOutputs, trainFeatures, maxIter0, gradTol0);
        const double pruneTol = 1e-3;
        slp.TruncateAlphaWeights(pruneTol);
    }
    // Perform more iterations on the reduced problem,
    // with the L1 penalty relaxed
    {
        const unsigned int maxIter1 = 100;
        const double gradTol1 = 1e-5;
        lfw.l1Alpha = 0.0;
        slp.SetLossFunctionWeights(lfw);
        ann::Train(slp, op, trainOutputs, trainFeatures, maxIter1, gradTol1);
    }
}

//!
//! Perform a final sweep optimization of the network
//!
void NetworkOptimizationFinalSweep(
    ann::SingleLayerPerceptron& slp,
    dvec& trainOutputs, 
    utilities::matrix<double> trainFeatures)
{
    utilities::optimize::LBFGS op;
    ann::LossFunctionWeights lfw;
    SetResidualWeights(lfw, trainOutputs);
    const unsigned int maxIter = 3000;
    const double gradTol = 1e-10;
    lfw.l1Alpha = 0.0;
    slp.SetLossFunctionWeights(lfw);
    ann::Train(slp, op, trainOutputs, trainFeatures, maxIter, gradTol);
}

//!
//! Class container for loss logs
//!
class LossLog
{
    private:
    static const int _PYTHON_VERSION_ = 3;
    dvec minLossLog;    //!<    Log of min loss for a set of trials
    dvec maxLossLog;    //!<    Log of max loss for a set of trials
    dvec meanLossLog;   //!<    Log of mean loss for a set of trials
    dvec optLossLog;    //!<    Log of refined optimal loss
    
    public:
    //!
    //! Record a loss record
    //!
    void Record(
        const dvec& losses,     //!<    Loss functions for a group of first trials
        const double optLoss)   //!<    Optimial refined loss function
    {
        this->minLossLog.push_back(losses[0]);
        this->maxLossLog.push_back(losses[losses.size()-1]);
        {
            double sumLoss = 0;
            for(auto& it : losses)
            {
                sumLoss += it;
            }
            this->meanLossLog.push_back(sumLoss/losses.size());
        }
        this->optLossLog.push_back(optLoss);
    }
    
    //!
    //! Plot the currently stored loss records against a given variable
    //!
    void Plot(
        const std::string figName,
        const std::string label, 
        const std::vector<unsigned int>& dependent)
    {
        //  Output the log data to a temporary file
        std::ofstream f_out;
        std::string tempFileName = "./log_plot_data.tmp";
        f_out.open(tempFileName);
        if(!f_out.is_open())
        {
            std::cerr << " CANNOT CREATE FILE " << tempFileName <<std::endl;
            exit(EXIT_FAILURE);
        }
        f_out << dependent.size() << "\n";
        for(auto& it : dependent)
        {
            f_out << it << "\n";
        }
        for(auto& it : minLossLog)
        {
            f_out << it << "\n";
        }
        for(auto& it : maxLossLog)
        {
            f_out << it << "\n";
        }
        for(auto& it : meanLossLog)
        {
            f_out << it << "\n";
        }
        for(auto& it : optLossLog)
        {
            f_out << it << "\n";
        }
        if(f_out.is_open())
        {
            f_out.close();
        }
        //  Generate the python script that will perform the plotting
        std::stringstream pythonScript;
        pythonScript.str();
        pythonScript<<"#! //usr//bin//env python"<<_PYTHON_VERSION_<<"\n"
        "import matplotlib                              \n"
        "import numpy as np                             \n"
        "matplotlib.use('Agg')                        \n\n"
        "import matplotlib.pyplot as plt              \n\n"
        "log_data = np.genfromtxt(\""<<tempFileName<<"\")\n"
        "log_size = int(log_data[0])                     \n"
        "x = log_data[1:1+log_size]                     \n"
        "minLossLog = log_data[1+log_size:1+2*log_size] \n"
        "maxLossLog = log_data[1+2*log_size:1+3*log_size]\n"
        "meanLossLog = log_data[1+3*log_size:1+4*log_size]\n"
        "optLossLog = log_data[1+4*log_size:]           \n"
        "plt.plot(x, np.log(minLossLog), label='min')   \n"
        "plt.plot(x, np.log(maxLossLog), label='max')   \n"
        "plt.plot(x, np.log(meanLossLog), label='mean') \n"
        "plt.plot(x, np.log(optLossLog), label='opt')   \n"
        "plt.legend(loc='best')                         \n"
        "plt.title(\"Log loss function vs "<<label<<"\")\n"
        "plt.savefig(\""<<figName<<".pdf\", bbox_inches=\'tight\') \n"
        "plt.close() \n";
        //  Execute the script
        utilities::Script myScript;
        myScript.SetScript(pythonScript.str());
        myScript.Execute();
    }
};

int main(int argc, char *argv[])
{
    //  Import Hamiltonian data from a file
    utilities::matrix<double> features;
    dvec outputs;
    {
        Hamiltonian hamiltonian;
        hamiltonian.ImportTerms();
        hamiltonian.GetCoefficients(outputs);
        hamiltonian.BuildFeatures(features);
        
        for(auto it_test = outputs.begin();
            it_test < outputs.end();++it_test)
        {
            //std::cout << "\t" << *it_test << std::endl;
        }
        std::cout << "Any key to continue" << std::endl;
        getchar();
    }
    //  Define neural network parameters
    const double trainProportion = 1.0;
    const unsigned int nbrStarts = 100;
    const unsigned int nbrBestOutcomes = 5;
    const unsigned int N = outputs.size();
    const unsigned int trainN = trainProportion*N;
    const unsigned int testN = (1-trainProportion)*N;
    const unsigned int P = features.m_dSecond;
    const unsigned int Hmin = 2;
    const unsigned int Hmax = 20;
    std::vector<unsigned int> Hvalues;
    for(unsigned int H=Hmin; H<Hmax; ++H)
    {
        Hvalues.push_back(H);
    }
    //  Divide into training and testing sets
    utilities::matrix<double> trainFeatures(P, trainN);
    utilities::matrix<double> testFeatures(P, testN);
    dvec trainOutputs(trainN);
    dvec testOutputs(testN);
    GetTrainTestDivision(features, trainFeatures, testFeatures, 
                         outputs, trainOutputs, testOutputs);
    //  Run network optimization
    LossLog log;
    for(auto& H : Hvalues)
    {
        std::cout << "\n\tFitting network for H=" << H << std::endl;
        ann::SingleLayerPerceptron slp(P, H, "shifted-exp");
        slp.AllocateWork(trainN);
        //  Optimize network from multiple random starting
        //  weights with early stopping, then take a few of 
        //  the best outcome and optimize further
        const unsigned int startSeed = time(NULL);
        dvec losses(nbrStarts);
        std::vector<unsigned int> seeds(nbrStarts);
        for(unsigned int trialIndex=0, seed=startSeed; trialIndex<nbrStarts; ++trialIndex, ++seed)
        {
            NetworkOptimizationFirstSweep(slp, trainOutputs, trainFeatures, seed);
            losses[trialIndex] = slp.EvaluateSquaredLoss(trainOutputs, trainFeatures);
            seeds[trialIndex] = seed;
        }
        utilities::QuickSort<double, unsigned int, _ASCENDING_ORDER_>(losses.data(), 
            seeds.data(), nbrStarts);
        double optLoss = losses[0];
        for(unsigned int bestIndex=0; bestIndex<nbrBestOutcomes; ++bestIndex)
        {
            //  Reconstruct weights from first sweep optimization
            NetworkOptimizationFirstSweep(slp, trainOutputs, trainFeatures, seeds[bestIndex]);
            //  Perform finer optimization
            NetworkOptimizationFinalSweep(slp, trainOutputs, trainFeatures);
            double currLoss = slp.EvaluateSquaredLoss(trainOutputs, trainFeatures);
            if(currLoss < optLoss)
            {
                optLoss = currLoss;
            }
        }
        //  Keep logs of the loss function values
        log.Record(losses, optLoss);
        
        //  Compute and display results
        dvec networkOutputs(N);
        slp.Evaluate(networkOutputs, features);
        //std::cout << "\n\t" << "Predicted outputs:" << std::endl;
        //std::cout << "\t" << "Prediction" << " " << "Actual" << std::endl;
        //for(auto it_network = networkOutputs.begin(), it_test = outputs.begin();
        //    it_network < networkOutputs.end(); ++it_network, ++it_test)
        //{
        //    std::cout << "\t" << *it_network << " " << *it_test << std::endl;
        //}
        //getchar();
    }    
    log.Plot("hidden_nodes_vs_loss", "Nbr hidden nodes", Hvalues);
    return EXIT_SUCCESS;
}
