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

#ifndef _MPI_WRAPPED_HPP_INCLUDED_
#define _MPI_WRAPPED_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../general/dcmplx_type_def.hpp" 
#include "../general/cout_tools.hpp"
#include "../general/template_tools.hpp"
#include <mpi.h>
#include <iomanip>
#include <string.h>
#include <vector>
#include <chrono>
#include <cstdint>

namespace utilities
{
    ////////////////////////////////////////////////////////////////////////////////		
    //!	\brief	A struct of variables and functions that can be used 
    //! to pass MPI interface between functions
    //!
    //!	These functions use the mpi/h library.
    //!	Examples at https://computing.llnl.gov/tutorials/mpi/
    ////////////////////////////////////////////////////////////////////////////////
    struct MpiWrapper
    {
        std::chrono::high_resolution_clock::time_point m_wallTime;
                        //!<	Keep track of the time taken during the program run
                        //!
        int m_nbrProcs;	//!<	Number of processor cores available	
                        //!
        int m_id;		//!<	Processor core identification number
                        //!
        std::vector<std::string> m_hosts;  
						//!<    Array to store host names of all process on 
                        //!
        int m_firstTask;
						//!<	Lowest task index for a parallelized task
                        //!
        int m_lastTask;	
						//!<	Highest task index for a parallelized task
                        //!
        bool m_exitFlag;//!<	Flag to exit MPI process on all cores
                        //!
        MPI_Comm m_comm;//!<    MPI communicator
                        //!
        Cout* m_cout;   //!<    a Cout object pointer, to control output 
                        //!     verbosity levels
        bool m_coutExternal;
						//!<	Flag to keep track of whether to delete the 
                        //!     Cout object or not when the class destructor is
                        //!     called (avoiding multiple deletions of the object)
        ////////////////////////////////////////////////////////////////////////////////
        //! \brief    Default constructor
        //!
        //! (default to using the global interface COMM_WORLD)
        ////////////////////////////////////////////////////////////////////////////////
        MpiWrapper()
        :
            m_comm(MPI_COMM_WORLD),
            m_coutExternal(false)
        {
            m_cout = new Cout;  //  Construct a default version of the object
                                //  and set it to that all output will be displayed        
            m_cout->SetVerbosity(0);
        }
        ////////////////////////////////////////////////////////////////////////////////
        //! \brief    Constructor to redirect the class internal verbosity object 
        //! (m_cout) to an external instance. 
        ////////////////////////////////////////////////////////////////////////////////
        MpiWrapper(Cout& cout)
        :
            m_comm(MPI_COMM_WORLD),
            m_coutExternal(true)
        {
            m_cout = &cout;
        }
        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	Destructor. This function is designed to be executed when
        //! the program terminates (which you can do by implementing this class 
        //! globally)																			
        ////////////////////////////////////////////////////////////////////////////////
        ~MpiWrapper()
        {
            if(m_id==0)	// FOR THE MASTER NODE	//
            {
                m_cout->MainOutput()<<"\n\t\tPROGRAM TERMINATED ";
                TimeStamp();
                auto duration = std::chrono::high_resolution_clock::now() - m_wallTime;
                auto hours    = std::chrono::duration_cast<std::chrono::hours>(duration);
                auto minutes  = std::chrono::duration_cast<std::chrono::minutes>(duration-hours);
                auto seconds  = std::chrono::duration_cast<std::chrono::seconds>(duration-hours-minutes);
                auto millis   = std::chrono::duration_cast<std::chrono::milliseconds>(duration-hours-minutes-seconds);
                m_cout->MainOutput()<<"\t\tTIME ELAPSED "<<hours.count()<<" HOURS "<<minutes.count()<<" MINUTES "<<seconds.count()<<"."<<millis.count()<<" SECONDS.\n"<<std::endl;
                m_cout->MainOutput()<<"------------------------------------------------------------------------------------------"<<std::endl;
                m_cout->MainOutput()<<"------------------------------------------------------------------------------------------"<<std::endl;
            }
            if(!m_coutExternal)
            {
                delete m_cout;
            }
            MPI_Finalize();
            return;
        }
        //  Define MPI utility functions
        void TimeStamp() const;
        void DivideTasks(const int id, const int nbrTasks, int nbrProcs,
                         int* firstTask, int* lastTask, const bool display) const;
        void TriangularDivideTasks(const int id, const int nbrTasks, int nbrProcs,
                                   int* firstTask, int* lastTask, const bool display) const;
        void Init(int argc, char *argv[]);
        void Init(int argc, char *argv[],bool);
        void ExitFlagTest();
        void GenerateHostNameList();
        void Sync(std::string& buffer, int syncNode) const;

        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	A function to implement MPI Gather, but with different sizes of 
        //! buffer on each node (the standard MPI_Gather assumes the buffer sizes are
        //! identical on all nodes).
        ////////////////////////////////////////////////////////////////////////////////
        template<typename T>
        void Gather(
            T* sendBuffer,     //!<    Send buffer
            int sendCount,     //!<    Size of send buffer (can be different for each node)
            T* recvBuffer,     //!<    Receive buffer: MUST BE OF TOTAL sendCount dimension
                               //!     on the node where the data are sent to. Not addressed
                               //!     on other nodes.
            int recvCount,     //!<    Size of the recvBuffer
            int gatherId,      //!<    id of the processor where the data are to be gathered
            MPI_Comm comm,     //!<    mpi communicator
            MPI_Status status) //!<    mpi status
            const
        {
            int nbrProcs;
            int id;
            MPI_Comm_size(comm, &nbrProcs);
            MPI_Comm_rank(comm, &id);
            if(id == gatherId)
            {
                int cumulativeSize = 0;
                for(int i=0; i<nbrProcs; ++i)
                {
                    if(i != gatherId)
                    {
                        int recv;
                        //  Receive the buffer size first
                        MPI_Recv(&recv, 1, MPI_INT, i, 40+2*i, comm, &status);    
                        //  Receive the buffer contents
                        MPI_Recv(recvBuffer+cumulativeSize, recv, this->GetType<T>(), i, 40+2*i+1, comm, &status);
                        cumulativeSize += recv;
                    }
                    else
                    {
                        memcpy(recvBuffer+cumulativeSize, sendBuffer, sizeof(T)*sendCount);
                        cumulativeSize += sendCount;
                    }
                }
            }
            else
            {
                //  Send the buffer size first 
                MPI_Send(&sendCount, 1, MPI_INT, gatherId, 40+2*id, comm);
                //  Send the buffer contents
                MPI_Send(sendBuffer, sendCount, this->GetType<T>(), gatherId, 40+2*id+1, comm); 
            }
            MPI_Barrier(comm);
            return;
        }

        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	A function to implement MPI Scatter, but with different sizes of 
        //! buffer on each node (the standard MPI_Scatter assumes the buffer sizes are
        //! identical on all nodes).
        ////////////////////////////////////////////////////////////////////////////////
        template<typename T>
        void Scatter(
            T* sendBuffer,     //!<    Send buffer MUST BE OF TOTAL recvCount dimension
                               //!     on the node where the data are sent from. Not addressed
                               //!     on other nodes.
            int sendCount,     //!<    Size of send buffer 
            T* recvBuffer,     //!<    Receive buffer 
            int recvCount,     //!<    Size of the recvBuffer (can be different for each node)
            int scatterId,     //!<    id of the processor where the data are to be scattered
            MPI_Comm comm,     //!<    mpi communicator
            MPI_Status status) //!<    mpi status
            const
        {
            int nbrProcs;
            int id;
        
            MPI_Comm_size(comm, &nbrProcs);
            MPI_Comm_rank(comm, &id);
            if(id == scatterId)
            {
                int cumulativeSize = 0;
                for(int i=0; i<nbrProcs; ++i)
                {
                    if(i != scatterId)
                    {
                        int send;
                        //  Receive the buffer size first
                        MPI_Recv(&send, 1, MPI_INT, i, 40+2*i, comm, &status); 
                        //  Send the buffer contents
                        MPI_Send(sendBuffer+cumulativeSize, send, this->GetType<T>(), i, 40+2*i+1, comm);
                        cumulativeSize += send;
                    }
                    else
                    {
                        memcpy(recvBuffer, sendBuffer+cumulativeSize, sizeof(T)*recvCount);
                        cumulativeSize += recvCount;
                    }
                }
            }
            else
            {
                //  Send the buffer size first 
                MPI_Send(&recvCount, 1, MPI_INT, scatterId, 40+2*id, comm);
                //  Receive the buffer contents
                MPI_Recv(recvBuffer, recvCount, this->GetType<T>(), scatterId, 40+2*id+1, comm, &status);
            }
            MPI_Barrier(comm);
            return;
        }

        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	A function to implement MPI Reduce, but with different sizes of 
        //! buffer on each node (the standard MPI_Reduce assumes the buffer sizes are
        //! identical on all nodes).
        //!
        //! Note: if the send and recieve buffers are the same on the reduce node
        //! then copying the reduce node allocation is ignored.
        //!
        //! The function is currently set up to sum the array values sent back to the
        //! reduce node
        ////////////////////////////////////////////////////////////////////////////////
        template<typename T>
        void Reduce(
            T* reduceBuffer,   //!<    Buffer containing the input/output
            T* recvBuffer,     //!<    Buffer to received mpi messages from other nodes
            int bufferCount,   //!<    Size of both the recvBuffer and the sendBuffer
            int reduceId,      //!<    id of the processor where the data are to be reduced
            MPI_Comm comm,     //!<    mpi communicator
            MPI_Status status) //!<    mpi status
            const
        {
            int nbrProcs;
            int id;
            MPI_Comm_size(comm, &nbrProcs);
            MPI_Comm_rank(comm, &id);
            if(reduceBuffer == recvBuffer)
            {
                std::cerr<<"ERROR WITH mpi.Reduce - reduce and recv buffers must have different addresses!"<<std::endl;
                exit(EXIT_FAILURE);
            }
            if(id == reduceId)
            {
                //  Recieve all other data
                for(int i=0; i<nbrProcs; ++i)
                {
                    if(i != reduceId)
                    {
                        //  Receive the buffer contents
                        MPI_Recv(recvBuffer, bufferCount, this->GetType<T>(), i, 40+2*i+1, comm, &status);    
                        //  Add to the reduce buffer
                        for(int j=0; j<bufferCount; ++j)
                        {
                            reduceBuffer[j] += recvBuffer[j];
                        }
                    }
                }
            }
            else
            {
                //  Send the buffer contents
                MPI_Send(reduceBuffer, bufferCount, this->GetType<T>(), reduceId, 40+2*id+1, comm);
            }
            MPI_Barrier(comm);
            return;
        }
        
        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	A template function to convert a given c variable type
        //! into an appropriate MPI type.
        //!
        //! \return An MPI type determined by the template argument T. 
        //! Default return is MPI_CHAR.
        ////////////////////////////////////////////////////////////////////////////////
        template<typename T>
        MPI_Datatype GetType() const
        {
            //  Integer types
            if(is_same<T, uint64_t>::value)          return MPI_UINT64_T;
            if(is_same<T, unsigned long int>::value) return MPI_UNSIGNED_LONG;
            if(is_same<T, uint32_t>::value)          return MPI_UINT32_T;
            if(is_same<T, unsigned int>::value)      return MPI_UNSIGNED;
            if(is_same<T, uint16_t>::value)          return MPI_UINT16_T;
            if(is_same<T, unsigned short int>::value)return MPI_UNSIGNED_SHORT;
            if(is_same<T, uint8_t>::value)           return MPI_UINT8_T;
            if(is_same<T, int64_t>::value)           return MPI_INT64_T;
            if(is_same<T, long int>::value)          return MPI_LONG;
            if(is_same<T, int32_t>::value)           return MPI_INT32_T;
            if(is_same<T, int>::value)               return MPI_INT;
            if(is_same<T, int16_t>::value)           return MPI_INT16_T;
            if(is_same<T, short int>::value)         return MPI_SHORT;
            if(is_same<T, int8_t>::value)            return MPI_INT8_T;
            //  Real types
            if(is_same<T, double>::value)            return MPI_DOUBLE;
            if(is_same<T, float>::value)             return MPI_FLOAT;
            //  Complex types
            if(is_same<T, dcmplx>::value)            return MPI_DOUBLE_COMPLEX;
            //  Bool types
            if(is_same<T, bool>::value)              return MPI_C_BOOL;
            //  Char types
            if(is_same<T, unsigned char>::value)     return MPI_UNSIGNED_CHAR;
            if(is_same<T, char>::value)              return MPI_CHAR;
            //  Default return
            return MPI_CHAR;
        }

        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	A template function to synchronize a given buffer, using the GetType
        //! function to automatically set the correct type
        ////////////////////////////////////////////////////////////////////////////////
        template<typename T>
        void Sync(
            T* buffer,          //!<  Pointer to the broadcast buffer
            int bufferSize,     //!<  Size of the buffer
            int syncNode)       //!<  Node to synchronize the buffer with
            const
        {
            MPI_Bcast(buffer, bufferSize, this->GetType<T>(), syncNode, m_comm);
        }

        ////////////////////////////////////////////////////////////////////////////////
        //!	\brief	A function to MPI synchronize a std::vector
        ////////////////////////////////////////////////////////////////////////////////
        template <typename T>
        void Sync(
            std::vector<T>* buffer, //!<  Broadcast buffer
            int syncNode)           //!<  Node to synchronize the buffer with
            const
        {
            unsigned long int bufferSize = 0;
            if(syncNode == m_id)
            {
                bufferSize = buffer->size();
            }
            MPI_Bcast(&bufferSize, 1, this->GetType<unsigned long int>(), syncNode, m_comm);
            buffer->resize(bufferSize);
            this->Sync<T>(buffer->data(),bufferSize,syncNode);
        }
        //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    };
}   //  End namespace utilities
#endif
