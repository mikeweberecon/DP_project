#%%
import numpy as np
from types import SimpleNamespace


def next_n_10(par,n_i,n_j,ccps_i,ccps_j): 
    # 1) set up
    A = par.A                                                               # array([0,1,…,k-1])
    n_matrix = np.zeros((par.n_bar+1, par.n_bar+1))  
    n_left    = par.n_bar - (n_i + n_j)
    if n_left <= 0:
        # no capacity left, stay put
        n_matrix[n_i, n_j] = 1.0
        return n_matrix

    # 2) split the leftover into winner/loser shares
    w = (n_left + 1)//2    # ceil(n_left/2)
    l = n_left - w         # floor(n_left/2)  

    # 3) loop over every action-pair
    for ai in A:
        for aj in A:
            p_ij = ccps_i[ai] * ccps_j[aj]

            # 3a) feasible: build
            if ai + aj <= n_left:
                ni2 = n_i + ai
                nj2 = n_j + aj
                n_matrix[ni2, nj2] += p_ij
                continue

            # 3b) strict over-bid
            if ai > aj and (ai + aj > n_left):
                # i outbids j
                if aj <= l:
                    ni2 = n_i + (n_left - aj)
                    nj2 = n_j + aj
                else:
                    ni2 = n_i + w
                    nj2 = n_j + l
                n_matrix[ni2, nj2] += p_ij
                continue

            if aj > ai and (ai + aj > n_left):
                # j outbids i
                if ai <= l:
                    ni2 = n_i + ai
                    nj2 = n_j + (n_left - ai)
                else:
                    ni2 = n_i + l
                    nj2 = n_j + w
                n_matrix[ni2, nj2] += p_ij
                continue

            # 3c) exact tie & over-bid, coin toss
            # split the weight 50/50 into the two half-splits
            if (ai == aj) and (ai + aj > n_left):
                n_matrix[n_i + w, n_j + l] += 0.5 * p_ij
                n_matrix[n_i + l, n_j + w] += 0.5 * p_ij

    total = n_matrix.sum()
    #print(f"[DEBUG] n_matrix sum  = {total:.12f}")
    return n_matrix

    
par = SimpleNamespace(A = np.arange(6),n_bar = 5)
next_n_10(par, 1,  2, np.ones(6)/6, np.ones(6)/6).round(2)


#%%

def f_x(par,n_i,n_j,FC,ccps_i,ccps_j): 

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Given state (n_i, n_j, FC) and CCPs, return a 3D matrix
    of probabilities for (n_i', n_j', FC').
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # 2D transition over n-space
    n_matrix = next_n_10(par,n_i,n_j,ccps_i,ccps_j)

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
    
        
    total3 = f_matrix.sum()
    #print(f"[DEBUG] f_matrix at ({n_i},{n_j},{FC}) sums to {total3:.12f}")
    return f_matrix

def F_X(par,ccps_i,ccps_j): 
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
                F_matrix[n_i, n_j, fc] = f_x(par, n_i, n_j, fc, ccp_i, ccp_j)
    
    row_sums = F_matrix.sum(axis=(3,4,5))   # this collapses the last three dims
    #print("[DEBUG] row_sums:", row_sums)
    return F_matrix











    


