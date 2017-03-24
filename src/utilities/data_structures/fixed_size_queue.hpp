////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		The file implements a fixed size queue. The data structure implements
//!     a push operation that overwrites the oldest value added to the queue,
//!     without moving queue entries. 
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

#ifndef _FIXED_SIZE_QUEUE_HPP_INCLUDED_
#define _FIXED_SIZE_QUEUE_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../linear_algebra/dense_vector.hpp"
#include <numeric>

namespace utilities
{
    //!
    //! Class container implementing a fixed size queue. The data structure 
    //! implements a push operation that overwrites the oldest value 
    //! added to the queue, without moving queue entries. 
    //!
    template<typename T>
    class FixedSizeQueue
    {
        private:
        int m_pushCtr;              //!<    Count the number of pushes
        std::vector<T> m_container; //!<    Container for queue elements
        std::vector<int> m_memMap;  //!<    Container for memory map, allowing
                                    //!     pushes with minimal element moving
        public:
        //!
        //! Default ctor
        //!
        FixedSizeQueue()
            :
            m_pushCtr(0)
        {}
        
        //!
        //! Rest the push counter to 0
        //!
        void reset()
        {
            m_pushCtr = 0;
            std::iota(m_memMap.begin(), m_memMap.end(), 0);
        }
        
        //!
        //! Set the size of the queue
        //!
        void resize(
            const unsigned int M)
        {
            m_container.resize(M);
            m_memMap.resize(M);
            this->reset();
        }
        
        //!
        //! Set the size of the queue and set elements to given value
        //!
        void resize(
            const unsigned int M, 
            T& value)
        {
            m_container.resize(M, value);
            m_memMap.resize(M);
            this->reset();
        }
        
        //!
        //! Access queue elements
        //!
        T& operator[](const int i)
        {
            return m_container[m_memMap[i]];
        }
        
        //!
        //! Implement a memory-efficient push operation
        //!
        void push(
            T& newVal)
        {
            if(m_pushCtr < m_memMap.size())
            {
                m_container[m_pushCtr] = newVal;
            }
            else
            {
                const int oldestIndex = 0;
                const int youngestIndex = m_memMap.size()-1;
                m_container[m_memMap[oldestIndex]] = newVal;
                for(auto& it : m_memMap)
                {
                    if(youngestIndex == it)
                    {
                        it = oldestIndex;
                    }
                    else
                    {
                        ++it;
                    }
                }
            }
            ++m_pushCtr;
        }
    };
}   //  End namespace utilities
#endif
