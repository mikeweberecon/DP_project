import numpy as np

def utility_i(par, n_i, FC, a_i):
    '''Utility function u(n_i,a_i) = p * n_i - FC * a_i'''
    return par.p * n_i- FC * a_i

def Pi(par, n_i, FC): 
    '''Helper to return shape for error prevention. Function that runs function "utility_i" '''

    #print(utility_i(par, n_i, FC, par.A[None, None, None, :]).shape)

    return utility_i(par, n_i, FC, par.A[None, None, None, :])

def Eps(par, ccps_i):
    '''Epsilon: Taste shocks'''

    return np.exp(1) - par.sigma*np.log(ccps_i)

