////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file 
//!	 	This file contains a template algorithm for sorting lists containing 
//!     associated pairs of values. The list can be sorted into ascending or
//!     descending order. The list can also be partially sorted. Complex
//!     numbers are always compared via their absolute values. 
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

#ifndef _QUICK_SORT_HPP_INCLUDED_
#define _QUICK_SORT_HPP_INCLUDED_

///////     LIBRARY INCLUSIONS     /////////////////////////////////////////////
#include "../general/dcmplx_type_def.hpp"
#if _DEBUG_
#include "../general/debug.hpp"
#endif

enum order_t {_ASCENDING_ORDER_,_DESCENDING_ORDER_};
//!<    Allow the user to specify the ordering used by the sort function
//!     via an enum type to be passd as a template parameter

namespace utilities
{
    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains an empty implementation that will be specialized 
    //! to allow for different comparison functions. Note that a struct containing 
    //! the functions is used here because the c++ standard does not currently allow 
    //! for partial template function specialization. 
    //////////////////////////////////////////////////////////////////////////////////
    template <typename K, order_t O>
    struct Compare
    {
        static bool Value(K* const & keyA, K* const & keyB)
        {
            return O;
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains a partial specialization to allow for sorting in 
    //! ascending order.
    //////////////////////////////////////////////////////////////////////////////////
    template <typename K>
    struct Compare<K, _ASCENDING_ORDER_>
    {
        static bool Value(K* const & keyA, K* const & keyB)
        {
            return *keyA <= *keyB;
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains a partial specialization to allow for sorting in 
    //! descending order.
    //////////////////////////////////////////////////////////////////////////////////
    template <typename K>
    struct Compare<K, _DESCENDING_ORDER_>
    {
        static bool Value(K* const & keyA, K*const & keyB)
        {
            return *keyA >= *keyB;
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains a full specialization to allow for std::pairs
    //! in ascending order.
    //////////////////////////////////////////////////////////////////////////////////
    template <>
    struct Compare<std::pair<uint64_t, dcmplx>, _ASCENDING_ORDER_>
    {
        static bool Value(std::pair<uint64_t, dcmplx>* const & keyA, 
                          std::pair<uint64_t, dcmplx>* const & keyB)
        {
            return (keyA->first <= keyB->first) || ((keyA->first == keyB->first) && (abs(keyA->second) <= abs(keyB->second)));
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains a full specialization to allow for std::pairs
    //! in descending order.
    //////////////////////////////////////////////////////////////////////////////////
    template <>
    struct Compare<std::pair<uint64_t, dcmplx>, _DESCENDING_ORDER_>
    {
        static bool Value(std::pair<uint64_t, dcmplx>* const & keyA, 
                          std::pair<uint64_t, dcmplx>* const & keyB)
        {
            return (keyA->first >= keyB->first) || ((keyA->first == keyB->first) && (abs(keyA->second) >= abs(keyB->second)));
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains a full specialization to allow for sorting complex 
    //! types in ascending order.
    //////////////////////////////////////////////////////////////////////////////////
    template <>
    struct Compare<dcmplx, _ASCENDING_ORDER_>
    {
        static bool Value(dcmplx* const & keyA, dcmplx* const & keyB)
        {
            return abs(*keyA) <= abs(*keyB);
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A class template to contain a comparison function used by the sorting
    //! algorithm, built with some template metaprogramming tricks.
    //!
    //! This version contains a full specialization to allow for sorting complex 
    //! types in descending order.
    //////////////////////////////////////////////////////////////////////////////////
    template <>
    struct Compare<dcmplx, _DESCENDING_ORDER_>
    {
        static bool Value(dcmplx* const & keyA, dcmplx* const & keyB)
        {
            return abs(*keyA) >= abs(*keyB);
        }
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief A template for a swap function, to swap the contents of two memory 
    //! addresses. This version is used as an alternative to the std::swap function
    //! which does not quite have the correct behaviour for this application (since
    //! std:swap swaps the pointers, not the addresses pointed to, given the same
    //! set of arguments as with this Swap function).
    //////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    void Swap(T*& a, T*& b)
    {
        const T temp = *a;
        *a = *b;
        *b = temp;
        return;
    }

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief Sorts keys and values lists according to the keys list. Returns
    //! after the first nSort elements are sorted. Sorting is performed with
    //! an in-place quick sort algorithm, with the pivot chosen as the median
    //! of the first, middle and last element of the partition.
    //!
    //! Template arguments:
    //! K: key data type, used for sorting
    //! V: value data type, not used for sorting
    //! O: Select from _ASCENDING_ORDER_ or _DESCENDING_ORDER_
    //////////////////////////////////////////////////////////////////////////////////
    template <typename K, typename V, order_t O>
    void DoQuickSort(
        K* keyList,                 //!<    Address of sort key list
        V* valueList,               //!<    Address of value list
        const unsigned int& dim,    //!<    Dimension of list
        const unsigned int& nSort,  //!<    Number of elements to sort before returning
        unsigned int& maxSorted)    //!<    The value up to which the algorithm has now 
                                    //!     sorted the input arrays                    
    {
        if(dim==0 || dim==1)  return;
        if(maxSorted>nSort)    return;
        K* p_keyFirst  = keyList;
        K* p_keyMiddle = keyList+dim/2;
        K* p_keyLast   = keyList+dim-1;
        {
            V* p_valueFirst  = valueList;
            V* p_valueMiddle = valueList+dim/2;
            V* p_valueLast   = valueList+dim-1;
            if(Compare<K,O>::Value(p_keyLast, p_keyMiddle))
            {
                Swap(p_keyLast, p_keyMiddle);
                Swap(p_valueLast, p_valueMiddle);
            }

            if(Compare<K,O>::Value(p_keyMiddle, p_keyFirst))
            {
                Swap(p_keyMiddle, p_keyFirst);
                Swap(p_valueMiddle, p_valueFirst);
            }

            if(Compare<K,O>::Value(p_keyLast, p_keyMiddle))
            {
                Swap(p_keyLast, p_keyMiddle);
                Swap(p_valueLast, p_valueMiddle);
            }
        }
        if(dim==2 || dim==3)
        {
            maxSorted += dim;
            return;
        }
        K pivot = *p_keyMiddle;
        K* p_keyLeft   = keyList;
        V* p_valueLeft = valueList;
        K* p_keyRight   = keyList+dim-1;
        V* p_valueRight = valueList+dim-1;
        unsigned int partitionSize = dim; 
        while(p_keyRight > p_keyLeft)
        {
            while(Compare<K,O>::Value(p_keyLeft, &pivot) && p_keyLeft < p_keyLast)
            {
                p_keyLeft++;
                p_valueLeft++;
            }
            
            while(Compare<K,O>::Value(&pivot, p_keyRight) && p_keyRight > p_keyFirst)
            {
                p_keyRight--;
                p_valueRight--;
                partitionSize--;
            }

            if(p_keyRight > p_keyLeft)
            {
                Swap(p_keyRight, p_keyLeft);
                Swap(p_valueRight, p_valueLeft);
            }
        }
        DoQuickSort<K, V, O>(keyList, valueList, partitionSize, nSort, maxSorted);
        DoQuickSort<K, V, O>(keyList+partitionSize, valueList+partitionSize, dim-partitionSize, nSort,maxSorted);
        return;
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief Sorts keys and values lists according to the keys list and returns
    //! after nSort values are in the correct order. This function is a wrapper to
    //! call the top level of the DoPairedSort recursion.
    //!
    //! Template arguments:
    //! K: key data type, used for sorting
    //! V: value data type, not used for sorting
    //! O: Select from _ASCENDING_ORDER_ or _DESCENDING_ORDER_
    //////////////////////////////////////////////////////////////////////////////////
    template<typename K, typename V, order_t O>
    void PartialQuickSort(
        K* keyList,                 //!<    Address of first list
        V* valueList,               //!<    Address of second list
        const unsigned int dim,     //!<    Dimension of list
        const unsigned int nSort)   //!<    Number of elements to sort before returning
    {
        unsigned int maxSorted = 0;
        const unsigned int nSortActual = std::min(dim,nSort);
        DoQuickSort<K,V,O>(keyList, valueList, dim, nSortActual, maxSorted);
    };
    
    //!
    //!	Overload of the PartialQuickSort function in case where only a key list
    //! is available. In this case the value list is not required
    //!
    template<typename K, typename V, order_t O>
    void PartialQuickSort(
        K* keyList,                 //!<    Address of first list
        const unsigned int dim,     //!<    Dimension of list
        const unsigned int nSort)   //!<    Number of elements to sort before returning
    {
        unsigned int maxSorted = 0;
        const unsigned int nSortActual = std::min(dim, nSort);
        V* valueList = new V[dim];
        DoQuickSort<K, V, O>(keyList, valueList, dim, nSortActual, maxSorted);
        delete[] valueList;
    };

    //////////////////////////////////////////////////////////////////////////////////
    //!	\brief Sorts keys and values lists according to the keys list. This
    //! function is a wrapper to call the top level of the DoPairedSort recursion.
    //!
    //! Template arguments:
    //! K: key data type, used for sorting
    //! V: value data type, not used for sorting
    //! O: Select from _ASCENDING_ORDER_ or _DESCENDING_ORDER_
    //////////////////////////////////////////////////////////////////////////////////
    template<typename K, typename V, order_t O>
    void QuickSort(
        K* keyList,                 //!<    Address of first list
        V* valueList,               //!<    Address of second list
        const unsigned int dim)     //!<    Dimension of list
    {
        unsigned int maxSorted = 0;
        DoQuickSort<K, V, O>(keyList, valueList, dim, dim, maxSorted);
    };

    //!
    //!	Overload of the QuickSort function in case where only a key list
    //! is available. In this case the value list is not required
    //!
    template<typename K, typename V, order_t O>
    void QuickSort(
        K* keyList,                 //!<    Address of first list
        const unsigned int dim)     //!<    Dimension of list
    {
        unsigned int maxSorted = 0;
        V* valueList = new V[dim];
        for(unsigned int i=0; i<dim; ++i)
        {
            valueList[i] = 0;
        }
        DoQuickSort<K,V,O>(keyList, valueList, dim, dim, maxSorted);
        delete[] valueList;
    };
}   //  End namespace utilities
#endif
