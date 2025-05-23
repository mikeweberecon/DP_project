#%%
import numpy as np
from types import SimpleNamespace

def next_n_bin(par,n_i,n_j,ccps_i,ccps_j):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Compute the 2D transition matrix for (n_i, n_j) → (n_i', n_j')
    given each player's CCPs and the remaining capacity.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Capacity left
    n_left = par.n_bar - n_i - n_j 

    # Initializing matrix of next state probabilities
    n_matrix = np.zeros((par.n_vals,par.n_vals))

    # If no capacity remains, stay in the same state with prob=1
    if n_left == 0: 

        n_matrix[n_i,n_j] = 1.0                                              # prob(n_next = n_now | n_now) = 1
        return n_matrix 
    
    # ——————————————————————————————————————————
    # Build joint action probabilities
    #   ccps_i[:, None]  → shape (n_vals,1)
    #   ccps_j[None, :]  → shape (1,n_vals)
    # Their product broadcasts to (n_vals,n_vals).
    # ——————————————————————————————————————————
    n_matrix = ccps_i[:,None]*ccps_j[None,:]                                           


    # Action index grids
    a_i,a_j = np.indices(n_matrix.shape)                                    #makes 2 2x2 matrices for actions of i and j 

    # Next-period counts 
    n_i_next = n_i + a_i
    n_j_next = n_j + a_j

    # Identify infeasible moves and ties 
    not_feasible = n_i_next + n_j_next > par.n_bar
    coin_toss = (a_i == a_j)*not_feasible

    # Multiplying coin toss probabilities in matrix 
    prob_sum_coin = np.sum(n_matrix[coin_toss])                             #next state prob in coin toss e cas(index [1,1])
    winner_n, loser_n = par.n_bar, 0 
    n_matrix[winner_n,loser_n] += 0.5*prob_sum_coin                         #transfering prob from index [1,1] to index [1,0]
    n_matrix[loser_n,winner_n] += 0.5*prob_sum_coin                         #transfering prob from index [1,1] to index [0,1]

    # setting probabilities to 0 to unfeasible next states 
    n_matrix[not_feasible] = 0 
    total = n_matrix.sum()
    print(f"[DEBUG] transition matrix sum at state = {total:.12f}")
    print("n_matrix:", n_matrix)
    return n_matrix 



def f_x_bin(par,n_i,n_j,FC,ccps_i,ccps_j): 

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Given state (n_i, n_j, FC) and CCPs, return a 3D matrix
    of probabilities for (n_i', n_j', FC').
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # 2D transition over n-space
    n_matrix = next_n_bin(par,n_i,n_j,ccps_i,ccps_j)

    # Build FC-transition vector
    p_fc = np.zeros(par.FC_vals)
    if FC > 0 : 
        p_fc[FC-1] = par.p_FC       #go down one index
        p_fc[FC]   = 1 - par.p_FC  #stay at the same
    else: 
        p_fc[0] = 1.0

    # Combine into 3D: shape = (n_vals, n_vals, FC_vals)
    f_matrix = n_matrix[:,:,None]*p_fc[None,None,:]   
                                                                                                                                            #print("ccps_i sum:",   np.sum(ccps_i),   "  any NaN?", np.isnan(ccps_i).any())
                                                                                                                                            #print("ccps_j sum:",   np.sum(ccps_j),   "  any NaN?", np.isnan(ccps_j).any())
                                                                                                                                            #print("ccps_i min/max:", np.nanmin(ccps_i), np.nanmax(ccps_i))
                                                                                                                                            #print("ccps_j min/max:", np.nanmin(ccps_j), np.nanmax(ccps_j))
    
        
    #total3 = f_matrix.sum()
                                                                            #print(f"[DEBUG] f_matrix at ({n_i},{n_j},{FC}) sums to {total3:.12f}")
    return f_matrix

def F_X_bin(par,ccps_i,ccps_j): 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Build the full 6D transition tensor F[(n_i,n_j,FC), (n_i',n_j',FC')]
    by iterating over all current states.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #1st,2nd,3rd D: current (n_i, n_j, fc)
    #4th, 5th, 6th D: given current state, probability of going to state (n_i',n_j',fc')
    # Pre-allocate: dims = (n_vals, n_vals, FC_vals, n_vals, n_vals, FC_vals)
    shape_F = (par.n_vals, par.n_vals, par.FC_vals)*2

    F_matrix = np.zeros(shape_F) + np.nan
    #for all states compute the ccps_i and ccps_j and then assign that to the next states function to get the full 6D matrix
    for n_i in range(par.n_vals):
        for n_j in range(par.n_vals):
            for fc in range(par.FC_vals):
                ccp_i = ccps_i[n_i, n_j, fc]
                ccp_j = ccps_j[n_i, n_j, fc]
                F_matrix[n_i, n_j, fc] = f_x_bin(par, n_i, n_j, fc, ccp_i, ccp_j)
    
                                                                                                                                            #row_sums = F_matrix.sum(axis=(3,4,5))   # this collapses the last three dims
                                                                                                                                            #print("[DEBUG] row_sums:", row_sums)
    return F_matrix
