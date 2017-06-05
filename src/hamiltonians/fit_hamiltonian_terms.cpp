////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!     This program reads in a set of coefficieints for Hamiltonian terms,
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
#include "hamiltonian_read_in.hpp"
#include "../neural_network/single_layer_perceptron.hpp"
#include "../neural_network/train.hpp"
#include "../neural_network/optimization_log.hpp"
#include "../utilities/general/dvec_def.hpp"
#include "../utilities/linear_algebra/dense_matrix.hpp"
#include "../utilities/optimization/lbfgs.hpp"
#include "../utilities/algorithms/quick_sort.hpp"
#include "../utilities/general/cout_tools.hpp"
#include "../utilities/general/load_bar.hpp"
#include "../utilities/wrappers/mpi_wrapper.hpp"
#include "../utilities/wrappers/program_options_wrapper.hpp"
#include "../program_options/general_options.hpp"

///////     GLOBAL DATA STRUCTURES      ////////////////////////////////////////
utilities::Cout utilities::cout;
utilities::MpiWrapper mpi(utilities::cout);

namespace myOptions
{
    namespace po = boost::program_options;
    po::options_description GetOptimizationOptions()
    {
        po::options_description optimizationOpt("Optimization options. Optimization proceeds with \n1. A first rought sweep with a large L1 penalty on alpha weights for a large set of random starting positions \n2. A second sweep with small alpha weights truncated. \n3. A refining sweep on a set of the top performers from the random set used in 1 and 2. ");
        optimizationOpt.add_options()
        ("min-h", po::value<unsigned int>()->default_value(5), 
         "Minimum number of hidden nodes to test.")
        ("max-h", po::value<unsigned int>()->default_value(20), 
         "Maximum number of hidden nodes to test.")
        ("zero-output-weight", po::value<double>()->default_value(10.0),
         "Set the weight to be given to the residuals of outputs that should be exactly zero; used when computing the loss function.")
        ("nbr-starts", po::value<unsigned int>()->default_value(100), 
         "Number of initial random starting parameters to test.")
        ("first-max-iter", po::value<unsigned int>()->default_value(20), 
         "Max LBFGS iterations allowed for the first sweep.")
        ("first-grad-tol", po::value<double>()->default_value(1e-2), 
         "LBFGS termination gradient tolerence for the first sweep.")
        ("first-l1", po::value<double>()->default_value(1.0), 
         "L1 penalty multiplier on alpha weights in the first sweep.")
        ("first-l2", po::value<double>()->default_value(0.0), 
         "L2 penalty multiplier on alpha weights in the first sweep.")
        ("prune-tol", po::value<double>()->default_value(1e-2),
         "Tolerence for pruning alpha weights after the first sweep.")
        ("second-max-iter", po::value<unsigned int>()->default_value(100), 
         "Max LBFGS iterations allowed for the second sweep.")
        ("second-grad-tol", po::value<double>()->default_value(1e-5), 
         "LBFGS termination gradient tolerence for the second sweep.")
        ("second-l1", po::value<double>()->default_value(0.0), 
         "L1 penalty multiplier on alpha weights in the second sweep.")
        ("second-l2", po::value<double>()->default_value(0.0), 
         "L2 penalty multiplier on alpha weights in the second sweep.")
        ("nbr-best-outcomes", po::value<unsigned int>()->default_value(5), 
         "Number of top first/second-sweep cases to further refine.")
        ("refine-max-iter", po::value<unsigned int>()->default_value(3000), 
         "Max LBFGS iterations allowed for the refining sweep.")
        ("refine-grad-tol", po::value<double>()->default_value(1e-10), 
         "LBFGS termination gradient tolerence for the refining sweep.")
        ("refine-l1", po::value<double>()->default_value(0.0), 
         "L1 penalty multiplier on alpha weights in the refining sweep.")
        ("refine-l2", po::value<double>()->default_value(0.0), 
         "L2 penalty multiplier on alpha weights in the refining sweep.");
        return optimizationOpt;
    };
}   //  End namespace myOptions

//!
//! Container for all optimization parameters
//!
struct OptimizationParameters
{
    unsigned int minH;          //!<    Smallest value of nbr Hidden nodes tested
    unsigned int maxH;          //!<    Largest value of nbr Hidden nodes tested
    double zeroOutputWeight;    //!<    Weight to be applied to the residuals of 
                                //!<    zero-outputs.
    unsigned int nbrStarts;     //!<    Number of random starts to consider
    unsigned int maxIter1;      //!<    Max LBFGS iterations in the first sweep
    double gradTol1;            //!<    LBFGS termination gradient tolerence in first sweep
    double firstL1;             //!<    First sweep L1 penalty on alpha weights
    double firstL2;             //!<    First sweep L2 penalty on alpha weights
    double pruneTol;            //!<    Tolerence for pruning alpha weights after first sweep
    unsigned int maxIter2;      //!<    Max LBFGS iterations in the second sweep
    double gradTol2;            //!<    LBFGS termination gradient tolerence in second sweep
    double secondL1;            //!<    Second sweep L1 penalty on alpha weights
    double secondL2;            //!<    Second sweep L2 penalty on alpha weights
    unsigned int nbrBestOutcomes;//!<   Number of top cases to refine further
    unsigned int maxIterR;      //!<    Max LBFGS iterations in the refining sweep
    double gradTolR;            //!<    LBFGS termination gradient tolerence in refining sweep
    double refiningL1;          //!<    Refining sweep L1 penalty on alpha weights
    double refiningL2;          //!<    Refining sweep L2 penalty on alpha weights
    
    //!
    //! Default ctor
    //!
    OptimizationParameters()
        :
        minH(5),
        maxH(20),
        zeroOutputWeight(10.0),
        nbrStarts(100),
        maxIter1(20),
        gradTol1(1e-2),
        firstL1(1.0),
        firstL2(0.0),
        pruneTol(1e-2),
        maxIter2(100),
        gradTol2(1e-5),
        secondL1(0.0),
        secondL2(0.0),
        nbrBestOutcomes(5),
        maxIterR(3000),
        gradTolR(1e-10),
        refiningL1(0.0),
        refiningL2(0.0)
    {}
    
    //!
    //! Ctor from command line
    //!
    OptimizationParameters(
    boost::program_options::variables_map* optionList,
                                        //!<    Parsed command line argument list
    utilities::MpiWrapper& mpi)         //!<    Address of the mpi wrapper class
    {
        GetOption(optionList, minH, "min-h", _LINE_, mpi);
        GetOption(optionList, maxH, "max-h", _LINE_, mpi);
        GetOption(optionList, zeroOutputWeight, "zero-output-weight", _LINE_, mpi);
        GetOption(optionList, nbrStarts, "nbr-starts", _LINE_, mpi);
        GetOption(optionList, maxIter1, "first-max-iter", _LINE_, mpi);
        GetOption(optionList, gradTol1, "first-grad-tol", _LINE_, mpi);
        GetOption(optionList, firstL1, "first-l1", _LINE_, mpi);
        GetOption(optionList, firstL2, "first-l2", _LINE_, mpi);
        GetOption(optionList, pruneTol, "prune-tol", _LINE_, mpi);
        GetOption(optionList, maxIter2, "second-max-iter", _LINE_, mpi);
        GetOption(optionList, gradTol2, "second-grad-tol", _LINE_, mpi);
        GetOption(optionList, secondL1, "second-l1", _LINE_, mpi);
        GetOption(optionList, secondL2, "second-l2", _LINE_, mpi);
        GetOption(optionList, nbrBestOutcomes, "nbr-best-outcomes", _LINE_, mpi);
        GetOption(optionList, maxIterR, "refine-max-iter", _LINE_, mpi);
        GetOption(optionList, gradTolR, "refine-grad-tol", _LINE_, mpi);
        GetOption(optionList, refiningL1, "refine-l1", _LINE_, mpi);
        GetOption(optionList, refiningL2, "refine-l2", _LINE_, mpi);
    }
    
    //!
    //! Mpi synchronize container contents with the specified node
    //!
    void MpiSync(
        int syncNode,                       //!<    Node to sync with
        utilities::MpiWrapper& mpi)         //!<    Address of the mpi wrapper class
    {
        mpi.Sync(&minH, 1, syncNode);
        mpi.Sync(&maxH, 1, syncNode);
        mpi.Sync(&zeroOutputWeight, 1, syncNode);
        mpi.Sync(&nbrStarts, 1, syncNode);
        mpi.Sync(&maxIter1, 1, syncNode);
        mpi.Sync(&gradTol1, 1, syncNode);
        mpi.Sync(&firstL1, 1, syncNode);
        mpi.Sync(&firstL2, 1, syncNode);
        mpi.Sync(&pruneTol, 1, syncNode);
        mpi.Sync(&maxIter2, 1, syncNode);
        mpi.Sync(&gradTol2, 1, syncNode);
        mpi.Sync(&secondL1, 1, syncNode);
        mpi.Sync(&secondL2, 1, syncNode);
        mpi.Sync(&nbrBestOutcomes, 1, syncNode);
        mpi.Sync(&maxIterR, 1, syncNode);
        mpi.Sync(&gradTolR, 1, syncNode);
        mpi.Sync(&refiningL1, 1, syncNode);
        mpi.Sync(&refiningL2, 1, syncNode);
    }
};

///////		FUNCTION FORWARD DECLARATIONS		    ////////////////////////////
boost::program_options::variables_map ParseCommandLine(int argc, char *argv[], 
                                                       utilities::MpiWrapper& mpi);
void SetResidualWeights(ann::LossFunctionWeights& lfw, dvec& outputs, 
                        OptimizationParameters& params);
void NetworkOptimizationFirstSweep(ann::SingleLayerPerceptron& slp,
    dvec& trainOutputs, utilities::matrix<double> trainFeatures,
    unsigned int seed, OptimizationParameters& params);
void NetworkOptimizationFinalSweep(ann::SingleLayerPerceptron& slp,
    dvec& trainOutputs, utilities::matrix<double> trainFeatures, 
    OptimizationParameters& params);

int main(int argc, char *argv[])
{
    mpi.Init(argc, argv);
    boost::program_options::variables_map optionList;
    optionList = ParseCommandLine(argc, argv, mpi);
    //  Import Hamiltonian data from a file
    utilities::matrix<double> features;
    dvec outputs;
    if(0 == mpi.m_id)	// FOR THE MASTER NODE
	{
        HamiltonianReadIn hamiltonian(&optionList, mpi);
        hamiltonian.ImportTerms(mpi);
        hamiltonian.GetCoefficients(outputs);
        hamiltonian.BuildFeatures(features);
    }
    MPI_Barrier(mpi.m_comm);
    mpi.ExitFlagTest();
    // Synchronize Hamiltonian data with the master node
    features.MpiSync(0, mpi);
    utilities::MpiSyncVector(outputs, 0, mpi);
    // Define neural network parameters
    OptimizationParameters params;
    if(0 == mpi.m_id)	// FOR THE MASTER NODE
	{
	    OptimizationParameters cmdParams(&optionList, mpi);
	    params = cmdParams;
	}
	mpi.ExitFlagTest();
	params.MpiSync(0, mpi);
	//  Divide into training and testing sets
    const double trainProportion = 1.0;
    const unsigned int N = outputs.size();
    const unsigned int trainN = trainProportion*N;
    const unsigned int testN = (1-trainProportion)*N;
    const unsigned int P = features.m_dSecond;
    utilities::matrix<double> trainFeatures(P, trainN);
    utilities::matrix<double> testFeatures(P, testN);
    dvec trainOutputs(trainN);
    dvec testOutputs(testN);
    ann::GetTrainTestDivision(features, trainFeatures, testFeatures, 
                              outputs, trainOutputs, testOutputs);
    //  Run network optimization
    std::vector<unsigned int> Hvalues;
    for(unsigned int H=params.minH; H<params.maxH; ++H)
    {
        Hvalues.push_back(H);
    }
    OptimizationLog lossLog;
    OptimizationLog nzWeightsPropLog;
    for(auto& H : Hvalues)
    {
        if(0 == mpi.m_id)	// FOR THE MASTER NODE
	    {
            utilities::cout.MainOutput() << "\n\tFitting network for H=" << H << std::endl;
        }
        ann::SingleLayerPerceptron slp(P, H, "shifted-exp");
        slp.AllocateWork(trainN);
        //  Optimize network from multiple random starting
        //  weights with early stopping
        dvec losses;
        std::vector<unsigned int> seeds;
        dvec nzWeightsProp;
        if(0 == mpi.m_id)	// FOR THE MASTER NODE
	    {
	        losses.resize(params.nbrStarts);
            seeds.resize(params.nbrStarts);
            nzWeightsProp.resize(params.nbrStarts);
	    }
	    mpi.DivideTasks(mpi.m_id, params.nbrStarts, mpi.m_nbrProcs, &mpi.m_firstTask, 
                        &mpi.m_lastTask, false);
        {
            int nbrTasks = mpi.m_lastTask - mpi.m_firstTask + 1;
            dvec nodeLosses(nbrTasks);
            std::vector<unsigned int> nodeSeeds(nbrTasks);
            dvec nodeNzWeightsProp(nbrTasks);
            unsigned int startSeed = 0;
            if(0 == mpi.m_id)	// FOR THE MASTER NODE
	        {
                startSeed = time(NULL);
            }
            mpi.Sync(&startSeed, 1, 0);
            //  Parallelize nbrStarts random initial optimizations
            utilities::LoadBar progressMeter;
            if(0 == mpi.m_id)	// FOR THE MASTER NODE
	        {
	            utilities::cout.AdditionalInfo() << "\n\tFirst pass:" << std::endl;
	            progressMeter.Initialize(nbrTasks);
	        }
            for(int trialIndex=0, seed=startSeed+mpi.m_firstTask; 
                trialIndex<nbrTasks; ++trialIndex, ++seed)
            {
                NetworkOptimizationFirstSweep(slp, trainOutputs, trainFeatures, seed, params);
                nodeLosses[trialIndex] = slp.EvaluateSquaredLoss(trainOutputs, trainFeatures);
                nodeSeeds[trialIndex] = seed;
                nodeNzWeightsProp[trialIndex] = (double)slp.nnzAlpha()/slp.countAlphaWeights();
                if(0 == mpi.m_id)	// FOR THE MASTER NODE
	            {
	                progressMeter.Display(trialIndex+1);
	            }
            }
            //  Mpi gather before sorting
            MPI_Status status;
            mpi.Gather(nodeLosses.data(), nodeLosses.size(), losses.data(), 
                       params.nbrStarts, 0, mpi.m_comm, status);
            mpi.Gather(nodeSeeds.data(), nodeSeeds.size(), seeds.data(), 
                       params.nbrStarts, 0, mpi.m_comm, status);
            mpi.Gather(nodeNzWeightsProp.data(), nodeNzWeightsProp.size(), nzWeightsProp.data(), 
                       params.nbrStarts, 0, mpi.m_comm, status);     
            if(0 == mpi.m_id)	// FOR THE MASTER NODE
	        {
                utilities::QuickSort<double, unsigned int, _ASCENDING_ORDER_>(losses.data(), 
                    seeds.data(), params.nbrStarts);
                utilities::QuickSort<double, unsigned int, _ASCENDING_ORDER_>(nzWeightsProp.data(), 
                    params.nbrStarts);
            }
        }
        //  Take a number of the best outcomes and perform a more refined optimization
        dvec refinedLosses;
        std::vector<unsigned int> topSeeds;
        dvec refinedNzWeightsProp;
        double optLoss = 0;
        double optNzWeightsProp = 0;
        if(0 == mpi.m_id)	// FOR THE MASTER NODE
        {
            refinedLosses.resize(params.nbrBestOutcomes);
            topSeeds.resize(params.nbrBestOutcomes);
            for(auto it_seeds=seeds.begin(), it_top=topSeeds.begin();
                it_top < topSeeds.end(); ++it_seeds, ++it_top)
            {
                *it_top = *it_seeds;
            }
            refinedNzWeightsProp.resize(params.nbrBestOutcomes);
        }
        mpi.DivideTasks(mpi.m_id, params.nbrBestOutcomes, mpi.m_nbrProcs, &mpi.m_firstTask, 
                        &mpi.m_lastTask, false);
        {
            int nbrTasks = mpi.m_lastTask - mpi.m_firstTask + 1;
            dvec nodeLosses(nbrTasks);
            std::vector<unsigned int> nodeSeeds(nbrTasks);
            dvec nodeNzWeightsProp(nbrTasks);
            //  Parallelize refined optimizations
            MPI_Status status;
            mpi.Scatter(topSeeds.data(), params.nbrBestOutcomes, nodeSeeds.data(), nbrTasks, 0, 
                        mpi.m_comm, status);
            utilities::LoadBar progressMeter;
            if(0 == mpi.m_id)	// FOR THE MASTER NODE
	        {
	            utilities::cout.AdditionalInfo() << "\n\tRefining pass:" << std::endl;
	            progressMeter.Initialize(nbrTasks);
	        }
            for(int bestIndex=0; bestIndex<nbrTasks; ++bestIndex)
            {
                //  Reconstruct weights from first sweep optimization
                NetworkOptimizationFirstSweep(slp, trainOutputs, trainFeatures, 
                    nodeSeeds[bestIndex], params);
                //  Perform finer optimization
                NetworkOptimizationFinalSweep(slp, trainOutputs, trainFeatures, params);
                nodeLosses[bestIndex] 
                    = slp.EvaluateSquaredLoss(trainOutputs, trainFeatures);
                nodeNzWeightsProp[bestIndex]
                    = (double)slp.nnzAlpha()/slp.countAlphaWeights();
                if(0 == mpi.m_id)	// FOR THE MASTER NODE
	            {
	                progressMeter.Display(bestIndex+1);
	            }
            }
            //  Mpi gather before sorting
            mpi.Gather(nodeLosses.data(), nodeLosses.size(), refinedLosses.data(), 
                       params.nbrBestOutcomes, 0, mpi.m_comm, status);
            mpi.Gather(nodeNzWeightsProp.data(), nodeNzWeightsProp.size(), refinedNzWeightsProp.data(), 
                       params.nbrStarts, 0, mpi.m_comm, status);   
            if(0 == mpi.m_id)	// FOR THE MASTER NODE
	        {
                utilities::QuickSort<double, unsigned int, _ASCENDING_ORDER_>(refinedLosses.data(), 
                    topSeeds.data(), params.nbrBestOutcomes);
                optLoss = refinedLosses[0];
                utilities::QuickSort<double, unsigned int, _ASCENDING_ORDER_>(refinedNzWeightsProp.data(), 
                    params.nbrBestOutcomes);
                optNzWeightsProp = refinedNzWeightsProp[0];
            }
        }
        //  Keep logs of the loss function values
        if(0 == mpi.m_id)	// FOR THE MASTER NODE
        {
            lossLog.Record(losses, optLoss);
            nzWeightsPropLog.Record(nzWeightsProp, optNzWeightsProp);
            //  Compute and display results
            #if 0
            dvec networkOutputs(N);
            slp.Evaluate(networkOutputs, features);
            std::cout << "\n\t" << "Predicted outputs:" << std::endl;
            std::cout << "\t" << "Prediction" << " " << "Actual" << std::endl;
            for(auto it_network = networkOutputs.begin(), it_test = outputs.begin();
            it_network < networkOutputs.end(); ++it_network, ++it_test)
            {
                std::cout << "\t" << *it_network << " " << *it_test << std::endl;
            }
            getchar();
            #endif
            utilities::cout.MainOutput() << std::endl;
        }
        MPI_Barrier(mpi.m_comm); 
    }
    if(0 == mpi.m_id)	// FOR THE MASTER NODE
	{  
        lossLog.Plot(true, "hidden_nodes_vs_loss", "nbr hidden nodes", "log loss function ", Hvalues);
        nzWeightsPropLog.Plot(false, "hidden_nodes_vs_nz_alphas", "nbr hidden nodes", "proportion nz alphas ", Hvalues);
    }
    return EXIT_SUCCESS;
}

//!
//! \brief A function to parse the command line arguments
//!
//! \return An instance of the boost program options variables map
//! containing the parsed command line arguments
//!
boost::program_options::variables_map ParseCommandLine(
    const int argc,                 //!<	Number of characters to parse
	char *argv[],	                //!<	Character array to parse
    utilities::MpiWrapper& mpi)     //!<    Instance of the mpi wrapper class
{
    namespace po = boost::program_options;
    po::variables_map vm;
    if(0 == mpi.m_id)	// FOR THE MASTER NODE
	{
	    //  Main program description
	    po::options_description allOpt("\n\tThis program attempts to fit selected groups of Hamiltonian terms with a specified neural network.\n\n\tThe program input options are as follows");
	    //	Declare option groups included
	    allOpt.add(myOptions::GetGeneralOptions()).add(myOptions::GetHamiltonianReadInOptions()).add(myOptions::GetOptimizationOptions());
	    try
        {
            po::store(po::command_line_parser(argc, argv).options(allOpt).run(), vm);
            if(vm.count("help"))
            {
	            utilities::cout.MainOutput() << allOpt << std::endl;
	            mpi.m_exitFlag = true;
            }
            po::notify(vm);
        }
        catch(po::error& e)
        {
            utilities::cout.MainOutput() << allOpt << std::endl;
            std::cerr << utilities::cout.HyphenLine() << std::endl;
            std::cerr << std::endl << "\tERROR:\t" << e.what() << std::endl;
            std::cerr << std::endl << utilities::cout.HyphenLine() << std::endl;
            mpi.m_exitFlag = true;
        }
        //  Extract the options specifying the length of positional options and then re-parse with 
        //  the positional options included
    }
    mpi.ExitFlagTest();
    //  Set global verbosity level
	if(0 == mpi.m_id)	// FOR THE MASTER NODE
	{
	    int verbosity;
	    GetOption(&vm, verbosity, "verbose", _LINE_, mpi);
	    utilities::cout.SetVerbosity(verbosity);
    }
    //  MPI sync verbosity level
    utilities::cout.MpiSync(0, mpi.m_comm);
    if(0 == mpi.m_id)	// FOR THE MASTER NODE
    {
        utilities::cout.MainOutput() << "\n\tRun with -h option to see program options" << std::endl;
    }
    return vm;
}

//!
//! Set loss function residual weights to penalize
//! more or less heavily the network outputs that
//! are expected to be exaclty zero
//!
void SetResidualWeights(
    ann::LossFunctionWeights& lfw,
    dvec& outputs,
    OptimizationParameters& params)
{
    dvec residualWeights(outputs.size());
    for(unsigned int index=0; index<outputs.size(); ++index)
    {
        if(outputs[index] == 0.0)
        {
            residualWeights[index] = params.zeroOutputWeight;
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
    unsigned int seed,
    OptimizationParameters& params)
{
    utilities::optimize::LBFGS op;
    ann::LossFunctionWeights lfw;
    SetResidualWeights(lfw, trainOutputs, params);
    //  Perform a few iterations with a heavy L1 penalty
    //  on the alpha weights, then prune small weights
    {
        lfw.l1Alpha = params.firstL1;
        lfw.l2Alpha = params.firstL2;
        slp.SetLossFunctionWeights(lfw);
        slp.RandomizeWeights(0.5, seed);
        ann::Train(slp, op, trainOutputs, trainFeatures, params.maxIter1, params.gradTol1);
        slp.TruncateAlphaWeights(params.pruneTol);
    }
    // Perform more iterations on the reduced problem,
    // with the L1 penalty relaxed
    {
        lfw.l1Alpha = params.secondL1;
        lfw.l2Alpha = params.secondL2;
        slp.SetLossFunctionWeights(lfw);
        ann::Train(slp, op, trainOutputs, trainFeatures, params.maxIter2, params.gradTol2);
    }
}

//!
//! Perform a final sweep optimization of the network
//!
void NetworkOptimizationFinalSweep(
    ann::SingleLayerPerceptron& slp,
    dvec& trainOutputs, 
    utilities::matrix<double> trainFeatures,
    OptimizationParameters& params)
{
    utilities::optimize::LBFGS op;
    ann::LossFunctionWeights lfw;
    SetResidualWeights(lfw, trainOutputs, params);
    lfw.l1Alpha = params.refiningL1;
    lfw.l2Alpha = params.refiningL2;
    slp.SetLossFunctionWeights(lfw);
    ann::Train(slp, op, trainOutputs, trainFeatures, params.maxIterR, params.gradTolR);
}
