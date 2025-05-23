import numpy as np
import aux
import bellman_operator
import ccp_operator

def objective_function(x, par):
    # unpack CCP guesses
    ccps_one_pre, ccps_two_pre = aux.prep_input(par, x) # x is a flat vector containing all ccps for both players. By running this function it converts it to 2 4D shape array. (ccps_one_pre, ccps_two_pre)

    # 1) Value‐function step
    EV_one_post = bellman_operator.bellman_operator_14(par, ccps_one_pre, ccps_two_pre)    #solve the reduced bellman operator for each player given this 4D shape of ccps
    EV_two_post = bellman_operator.bellman_operator_14(par, ccps_two_pre, ccps_one_pre)    

    # 2) CCP best‐response
    ccps_one_post = ccp_operator.compute_ccps_i(par, ccps_one_pre, ccps_two_pre, EV_one_post) #compute updated ccps given these value functions (Best response)
    ccps_two_post = ccp_operator.compute_ccps_i(par, ccps_two_pre, ccps_one_pre, EV_two_post)

    # 3) pack again to 1D
    x_out = aux.prep_output(par, ccps_one_post, ccps_two_post)   #takes the 4D array result and runs prep_output to flatten everything into 1 vector of ccps of both players. 
    
    return x_out 



def objective_for_root(x,par): 
    
    return objective_function(x,par) - x