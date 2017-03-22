////////////////////////////////////////////////////////////////////////////////
//!
//!                         \author Simon C. Davenport 
//!
//!  \file
//!		This program reads in a set of coefficieitns for Hamiltonain terms,
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
#include "../neutral_network/single_layer_perceptron.hpp"

int main(int argc, char *argv[])
{
    // Import test set of exact Hamiltonian terms
    
    // Generate a set of "features" used as inputs to the neutral network
    
    // Fit the neutral network using a certian optimization schedule. Real and
    // imaginary parts of the terms can be fitted separately.
    
    // Plot some fitting error metrics
    
    return EXIT_SUCCESS;
}