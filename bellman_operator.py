import numpy as np
import transitions
import transitions_binary
import utility

def bellman_operator_14(par,ccps_i,ccps_j):

    '''Computes the Bellman operator for player i using equation 14 from Egesdal, Su'''

    #1. Prepare state grids 
    n_i_grid = par.n_grid[:,None,None,None]       # Shape(n_vals, 1,1,1)
    FC_grid = par.FC_grid[None,None,:, None]      # Shape (1,1,FC_vals,1)

    #2. Calculate payoff components
    pis = utility.Pi(par, n_i_grid, FC_grid)      # Computing observable part of period utility shape(n_vals,1,FC_vals,k)
    eps = utility.Eps(par, ccps_i)                # Computing the taste shock  (same shape as "pis")

    #3. Calculate expected payoff
    payoff = pis + eps                          
    weighted_payoff = ccps_i*(payoff)                # Weight by CCPs
    sum_payoffs = np.sum(weighted_payoff,axis = -1)  # Sums payoffs over all actions of player i (last dimension)
    sum_payoffs_1D = sum_payoffs.ravel()             # Flatten to 1D

    #4. Reshape F_matrix to 2D for linear equation solving
    S = par.n_vals * par.n_vals * par.FC_vals
    F_matrix = transitions.F_X(par,ccps_i,ccps_j) # Shape (nvals, nvals, fcvals, nvals, nvals, fcvals) 
    F_matrix_2D = F_matrix.reshape(S,S)                      # Reshape to 2D matrix 


    #5. Solve the linear equation: (I- beta*F)*V = payoffs 
    I = np.eye(S)                                            # Produces and SxS identity matrix 
    multiplier = I - par.beta*F_matrix_2D                    # Matrix to invert
 
    # Solve for the Value function using this linear equation. 
    V_operator_vector = np.linalg.solve(multiplier,sum_payoffs_1D)            

    #6. Reshape the solution to 3D grid format  
    V_operator = V_operator_vector.reshape(par.n_vals,par.n_vals,par.FC_vals)

    return V_operator 


