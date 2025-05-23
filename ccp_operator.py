import numpy as np
import transitions
import transitions_binary
import utility

#######################
#VALUE CHOICE FUNCTION#
#######################

def value_choice(par, n_i, n_j, FC, F_X, EV_next):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Compute the value of choosing a specific action given the current state.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # In-period payoff: period utility + discounted expected future value 
    pis = utility.Pi(par, n_i, FC)
    future_value = par.beta*np.sum(F_X*EV_next)

    return pis + future_value 

def value_choice_total(par, ccps_i, ccps_j, EV_next):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Compute the value of each action for every state combination.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Obtain the full 6D transition matrix from binary transitions
    F_matrix = transitions.F_X(par, ccps_i, ccps_j)

    # Initialize container for value choices
    value_choices = np.zeros((par.n_vals, par.n_vals, par.FC_vals, par.k)) + np.nan

    # Loop over state space to compute value choices
    for n_i in range(par.n_vals):
        for n_j in range(par.n_vals):
            for FC in range(par.FC_vals):

                # Extract transition probabilities for this specific state
                F_state = F_matrix[n_i, n_j, FC]
        
                # Calculate value choices for the current state
                value_choices[n_i, n_j, FC] = value_choice(par, n_i, n_j, FC, F_state, EV_next)
                

    return value_choices


#######################
    #CCP OPERATOR#
#######################

def compute_ccps_i(par, ccps_i, ccps_j, EV_next):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Compute the Conditional Choice Probabilities (CCPs) for player i.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #1. Calculate the value of each choice for every state
    vc = value_choice_total(par, ccps_i, ccps_j, EV_next)

    #2. Subtract the maximum value for numerical stability (log-sum-exp trick)
    m     = np.max(vc, axis=-1, keepdims=True)      # shape (...,1)
    exp_v = np.exp(vc - m)                          # now at least one entry per row is >=1

    #3. Normalize to get probabilities (Softmax operation)
    denom = np.sum(exp_v, axis=-1, keepdims=True)   # strictly > 0 everywhere
    ccps  = exp_v / denom

    return ccps
