
import numpy as np


def input_vector_indx(par):

    N_P = par.n_vals*par.n_vals*par.FC_vals*par.k # Total elements in one P vector

    split_here = int(N_P)

    return split_here

def input_vector_shapes(par):

    P_shape = (par.n_vals, par.n_vals, par.FC_vals, par.k)

    return P_shape

def prep_input(par, input_vector):

    # takes 1D input-vector with two flat P vectors in one
    # returns two 4D P matrices

    split_here = input_vector_indx(par)
    P_shape = input_vector_shapes(par)

    P_one = input_vector[:split_here].reshape(P_shape)
    P_two = input_vector[split_here:].reshape(P_shape)

    return P_one, P_two

def prep_output(par, ccps_one, ccps_two):

    # takes two 4D P matrices
    # returns 1D input-vector with two flat P vectors in one

    split_here = input_vector_indx(par)

    N_P = par.n_vals*par.n_vals*par.FC_vals*par.k
    output_vector = np.zeros((2*N_P))

    output_vector[:split_here] = ccps_one.flatten()
    output_vector[split_here:] = ccps_two.flatten()

    return output_vector

def draw_guess(par):
        
    P_shape = input_vector_shapes(par)

    # draw and normalize ccp guesses
    ps_raw = np.random.uniform(0, 1, size=P_shape)
    ps = ps_raw/(np.sum(ps_raw, axis=-1)[..., None])

    input_vec = np.concatenate((ps.flatten(), ps.flatten()), axis=0)

    return input_vec

def check_new_eq(sol, new_sol_conj):

    # unpack known sols
    known_sol = sol.raw_sol

    # round
    new_sol = np.round(new_sol_conj, decimals=sol.dec).reshape(1, -1) # make as a row

    known_sol_rounded = np.round(known_sol, decimals=sol.dec) # shape (iterations, 2*k)

    # check
    match = np.all(new_sol == known_sol_rounded, axis=1) # check if all probabilities match for the specific rows -> returns shape (iterations,)

    if np.any(match): # if there are any matches, not new eq

        return False
    
    else: # if no matches, new eq

        return True
    
def print_eq(new_eq_bool, eq_counter):

    if new_eq_bool: 
        print(f'New eq found: Total number of eqa is {eq_counter}\n')
    
    else:
        print(f'No new eq found: Total number of eqa is {eq_counter}\n')