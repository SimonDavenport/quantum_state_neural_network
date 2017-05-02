////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file 
//!		This file contains a bunch of functions used for MPI programming
//!     Examples at https://computing.llnl.gov/tutorials/mpi/
//!
//!                    Copyright (C) Simon C Davenport
//!
//!		This program is free software: you can redistribute it and/or modify
//!		it under the terms of the GNU General Public License as published by
//!		the Free Software Foundation, either version 3 of the License,
//!		or (at your option) any later version.
//!
//!		This program is distributed in the hope that it will be useful, but
//!		WITHOUT ANY WARRANTY; without even the implied warranty of 
//!		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
//!		General Public License for more details.
//!
//!		You should have received a copy of the GNU General Public License
//!		along with this program. If not, see <http://www.gnu.org/licenses/>.
//! 
////////////////////////////////////////////////////////////////////////////////

#include "mpi_wrapper.hpp"

namespace utilities
{

//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    ////////////////////////////////////////////////////////////////////////////////
    //!	\brief	This function must be executed at the start of a program using MPI			
    //!																				
    //!	It will initialise the global variables nbrProces and id to contain the		
    //!	appropriate number of processors available and the identifier of the 		
    //!	running process. Also it will set the exit function to be Terminate
    //!																				
    ////////////////////////////////////////////////////////////////////////////////
    void MpiWrapper::Init(
        int argc,       //!<    Length of argument list      
        char *argv[])   //!<    Initialize from command line arguments
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(m_comm,&m_nbrProcs);
        MPI_Comm_rank(m_comm,&m_id);
        if(m_id==0)		// FOR THE MASTER NODE
        {
            m_cout->MainOutput()<<"\n\n\t\tPROGRAM COMMENCED ";
            TimeStamp();
            m_wallTime = std::chrono::high_resolution_clock::now();
            m_cout->MainOutput()<<"\t\tRUNNING ON "<<m_nbrProcs<<" NODE(S)"<<std::endl;
        }
        this->GenerateHostNameList();
        return;
    }

//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    ////////////////////////////////////////////////////////////////////////////////
    //!	\brief	This function must be executed at the start of a program using MPI			
    //!																				
    //!	It will initialise the global variables nbrProces and id to contain the		
    //!	appropriate number of processors available and the identifier of the 		
    //!	running process. Also it will set the exit function to be Terminate
    //!																				
    ////////////////////////////////////////////////////////////////////////////////
    void MpiWrapper::Init(
        int argc,       //!<    Length of argument list      
        char *argv[],   //!<    Initialize from command line arguments
        bool silent)    //!<    Flag to not print any output
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(m_comm,&m_nbrProcs);
        MPI_Comm_rank(m_comm,&m_id);
        if(m_id==0)		// FOR THE MASTER NODE
        {
            if(!silent)
            {
                m_cout->MainOutput()<<"\n\n\t\tPROGRAM COMMENCED ";
                TimeStamp();
                m_cout->MainOutput()<<"\t\tRUNNING ON "<<m_nbrProcs<<" NODES"<<std::endl;
            }
            
            m_wallTime = std::chrono::high_resolution_clock::now();
        }
        this->GenerateHostNameList();
        return;
    }

//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    ////////////////////////////////////////////////////////////////////////////////
    //! \brief  TimeStamp prints the current YMDHMS date
    //!
    ////////////////////////////////////////////////////////////////////////////////
    
    void MpiWrapper::TimeStamp() const
    {
        static char time_buffer[40];
        const struct tm *tm_ptr;
        time_t now;
        now = time(NULL);
        tm_ptr = localtime(&now);
        strftime(time_buffer, 40, "%d %B %Y %I:%M:%S %p", tm_ptr);
        m_cout->MainOutput()<<time_buffer<<"\n\n";
        return;
    }
    
//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    ////////////////////////////////////////////////////////////////////////////////
    //! \brief  GenerateHostNameList gets the host name of each processor
    //! and then combines into a single table stored on the master node
    //!
    ////////////////////////////////////////////////////////////////////////////////
    void MpiWrapper::GenerateHostNameList()
    {    
        const int bufferSize = 40;
        int variableSize = bufferSize;
        char host[bufferSize];
        MPI_Get_processor_name(host, &variableSize);
        char* recvBuffer=0;
        if(0 == m_id)   //  For the master node
        {
            recvBuffer = new (std::nothrow) char[m_nbrProcs*bufferSize];
        }
        MPI_Gather(host, bufferSize, MPI_CHAR, recvBuffer, bufferSize, MPI_CHAR, 0, m_comm);
        if(0 == m_id)   //  For the master node
        {
            for(int i=0; i<m_nbrProcs; ++i)
            {
                memcpy(host, recvBuffer+bufferSize*i, bufferSize*sizeof(char));
                m_hosts.push_back(host);
            }
        }
        if(0 != recvBuffer)
        {
            delete[] recvBuffer;
        }
        return;
    }
    
//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    ////////////////////////////////////////////////////////////////////////////////
    //!	\brief 	DivideTasks divides up nbrTasks tasks amongst nbrProcs processes
    //!	On the processor labelled by the identifier id, the function returns the
    //!	appropriate index of the minimum and maximum task number
    //!
    //!	Example output:	(arugments nbrTasks=450, nbrProcs=4)
    //!
    //!	\verbatim
    //!	Parallelization:
    //!
    //!				            Number of  	  First     Last
    //!	Processor    Host        Tasks        Task      Task
    //!											
    //!		0        pc1          113		   0		 112	
    //!		1	     pc1          112	      113	     224	
    //!		2	     pc2          113	      225	     337	
    //!		3	     pc2          112	      338	     449	
    //!	\endverbatim
    ////////////////////////////////////////////////////////////////////////////////
    void MpiWrapper::DivideTasks(
        const int id,			//!<	The unique identifier of a particular processor
        const int nbrTasks,	    //!<	The number of tasks to assign
        int nbrProcs,	        //!<	The number of processors available to assign tasks to
        int* firstTask,	        //!<	The unique index of the first task for processor id
        int* lastTask,			//!<	The unique index of the last task for processor id
        const bool display)     //!<    Flag to turn on/off command line display
        const
    {
        if(id==0)	// FOR THE MASTER NODE	//
        {
            if(display)
            {
                m_cout->AdditionalInfo() << "\n\tParallelization:\n"<<std::endl;
                m_cout->AdditionalInfo() << "\t                            Number of      First       Last\n";
                m_cout->AdditionalInfo() << "\tProcessor      Host          Tasks          Task        Task\n"<<std::endl;
            }
        }
        if(nbrTasks<nbrProcs)
        {
            nbrProcs = nbrTasks;
            if(id>=nbrProcs)
            {
                *firstTask = nbrTasks-1;
                *lastTask  = nbrTasks-2;
            }
        }
        int procRemain = nbrProcs;
        int taskRemain = nbrTasks;
        int hi,lo;
        hi = 0;
        for(int proc = 0; proc < nbrProcs; ++proc)
        {
            int taskProc = (int)floor((double)taskRemain/procRemain+0.5);
            procRemain--;
            taskRemain-=taskProc;
            lo=hi+1;
            hi+=taskProc;
            if(id==proc)
            {
                *firstTask=lo-1;
                *lastTask=hi-1;
            }
            if(id==0)	// FOR THE MASTER NODE	//
            {
                if(display && (int)m_hosts.size() >= proc)
                {
                    m_cout->AdditionalInfo()<<"\t"<<proc<<"\t"<<std::setw(10)<<m_hosts[proc]<<"\t"<<std::setw(12)<<taskProc<<std::setw(12)<< lo-1<<std::setw(12)<<hi-1<< "\n";
                }
            }
        }
        if(id==0)	// FOR THE MASTER NODE	//
        {
            if(display)
            {
                m_cout->AdditionalInfo()<<std::endl;
            }
        }
        return;
    }

//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    ////////////////////////////////////////////////////////////////////////////////
    //!	\brief 	TriangularDivideTasks divides up nbrTasks tasks amongst nbrProcs 
    //! processes with a trangular weighting function (used e.g. for allocating
    //! a roughly equal memory distribution to store a triagular matrix in parallel)
    //!
    //!	On the processor labelled by the identifier id, the function returns the
    //!	appropriate index of the minimum and maximum task number
    //!
    //!	Example output:		(arugments nbrTasks=450, nbrProcs=4)
    //!
    //!	\verbatim
    //!    Parallelization:
    //!
    //!                 Number of   First      Last
    //!    Processor     Tasks     Task       Task
    //!
    //!            0       60      0       59
    //!            1       71      60      130
    //!            2       93      131     223
    //!            3       226     224     449
    //!
    //!	\endverbatim
    ////////////////////////////////////////////////////////////////////////////////
    void MpiWrapper::TriangularDivideTasks(
        const int id,			//!<	The unique identifier of a particular processor
        const int nbrTasks,		//!<	The number of tasks to assign
        int nbrProcs,		    //!<	The number of processors available to assign tasks to
        int* firstTask,			//!<	The unique index of the first task for processor id
        int* lastTask,			//!<	The unique index of the last task for processor id
        const bool display)     //!<    Flag to turn on/off command line display
        const
    {
        if(id==0)	// FOR THE MASTER NODE	//
        {
            if(display)
            {
                m_cout->AdditionalInfo() << "\n\tParallelization:\n"<<std::endl;
                m_cout->AdditionalInfo() << "\t                            Number of      First       Last\n";
                m_cout->AdditionalInfo() << "\tProcessor      Host          Tasks          Task        Task\n"<<std::endl;
            }
        }
        if(nbrTasks<nbrProcs)
        {
            nbrProcs = nbrTasks;
            if(id>=nbrProcs)
            {
                *firstTask = nbrTasks-1;
                *lastTask  = nbrTasks-2;
            }
        }
        int procRemain = nbrProcs;
        int taskRemain = nbrTasks;
        double weight  = 0.5*taskRemain*(taskRemain+1.0);
        int hi,lo;
        hi = 0;
        for(int proc = 0; proc < nbrProcs; ++proc)
        {         
            double weightProc = floor(weight/(double)procRemain+0.5);
            //  Function here comes from solving a quadratic equation for a sum
            //  of triagular numbers to find the tasks to assign to the current
            //  node
            int taskProc = (int)floor(0.5 + taskRemain 
                              - 0.5*sqrt(1+4.0*(double)taskRemain*(1+(double)taskRemain)-8.0*weightProc));
            procRemain--;
            taskRemain-=taskProc;
            weight  = 0.5*taskRemain*(taskRemain+1.0);
            lo=hi+1;
            hi+=taskProc;
            if(id==proc)
            {
                *firstTask=lo-1;
                *lastTask=hi-1;
            }
            if(id==0)	// FOR THE MASTER NODE	//
            {
                if(display)
                {
                    m_cout->AdditionalInfo()<<"\t"<<proc<<"\t"<<std::setw(10)<<m_hosts[proc]<<"\t"<<std::setw(12)<<taskProc<<std::setw(12)<< lo-1<<std::setw(12)<<hi-1<< "\n";
                }
            }
        }
        if(id==0)	// FOR THE MASTER NODE	//
        {
            if(display)
            {
                m_cout->AdditionalInfo()<<std::endl;
            }
        }
        return;
    }

//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    
    ////////////////////////////////////////////////////////////////////////////////
    //!	\brief	If the exitflag is set to true on the master node, then 
    //!	call Terminate
    ////////////////////////////////////////////////////////////////////////////////

    void MpiWrapper::ExitFlagTest()
    {
        MPI_Barrier(m_comm);
        MPI_Bcast(&m_exitFlag, 1, MPI_C_BOOL, 0, m_comm);
        if(m_exitFlag)
        {
            exit(EXIT_FAILURE);
        }
        return;
    }
    
//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
        
    ////////////////////////////////////////////////////////////////////////////////
    //!	\brief	A function to MPI synchronize a string
    //! 
    ////////////////////////////////////////////////////////////////////////////////

    void MpiWrapper::Sync(
        std::string& buffer, //!<  Pointer to the broadcast buffer
        int syncNode)        //!<  Node to synchronize the buffer with
        const
    {
        int bufferSize;
        if(syncNode == m_id)
        {
            bufferSize = buffer.length();
        }
        this->Sync(&bufferSize, 1, syncNode);
        if(syncNode == m_id)     //  FOR NODE syncNode
        {
            char *charArray;
            charArray = (char*)buffer.c_str();
            for(int i=0; i<m_nbrProcs; ++i)
	        {
	            if(syncNode != i)
	            {
                    MPI_Send(charArray, bufferSize, this->GetType<char>(), i, 21, m_comm);       
                }
            }
        }
        else	// FOR ALL OHTER NODES
        {
            char charArray[bufferSize];
            MPI_Status status;
            MPI_Recv(charArray, bufferSize, this->GetType<char>(), syncNode, 21, m_comm, &status);
            buffer = charArray;
            buffer.resize(bufferSize);
        }
    }    
//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//    
}   //  End namespace utilities
