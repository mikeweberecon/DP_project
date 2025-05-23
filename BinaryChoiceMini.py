#%%
import numpy as np
import scipy.optimize as opt
import time
import aux
import solvers
import transitions
import pandas as pd 

from types import SimpleNamespace


class binary_choice_mini:
    def __init__(self):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.setup()

    def setup(self):
        par = self.par
        sol = self.sol

        # --- model parameters ---
        par.n_bar   = 1
        par.p      = 0.5
        par.beta   = 0.99
        par.sigma  = 1.0
        par.Nplayers = 2

        # fixed‐cost tech
        par.min_FC = 0
        par.max_FC = 10
        par.p_FC   = 0.45

        # action space
        par.k = par.n_bar + 1
        par.A = np.arange(par.k)

        # state‐space dims
        par.n_vals  = par.n_bar + 1
        par.n_grid  = np.arange(par.n_vals)
        par.FC_grid = np.arange(par.min_FC, par.max_FC+1)
        par.FC_vals = len(par.FC_grid)
        par.FC_indices = np.arange(par.FC_vals)
        par.Nstates = 3   # (n_i,n_j,FC)
        sol.reps = 100
        

        # solver settings
        sol.iterations = 20
        sol.dec        = 4
        # raw_sol: store up to `iterations` distinct equilibria
        N_P = aux.input_vector_indx(par)
        sol.raw_sol = np.zeros((sol.iterations, 2*N_P)) #shape is a 2D matrix, rows are iterations, and columns are ccps of i, and then ccps of j in one vector given each combination of states.
        


    def solve(self,method="root"):
        sol = self.sol
        if method == "root":
            self.sol.x = solvers.root_solver(self.par,sol)
        elif method == "policy":
            self.sol.x = solvers.policy_iteration_solver(self.par,sol)
        print("Solution:", self.sol.x)
        return self.sol.x
    

    
    def sim_data(self, N, T, seed=2020):
        np.random.seed(seed)
        par, sol = self.par, self.sol

        # 1) unpack CCPs (P_shape = (par.n_vals, par.n_vals, par.FC_vals, par.k)
        x_vec = sol.x.ravel()
        ccps_i, ccps_j = aux.prep_input(par, x_vec)    

        # 2) index matrix for flattening
        idx = np.tile(np.arange(1, N+1), (T, 1))   #creates a TxN matrix where each collumns are the Ns
        t   = np.tile(np.arange(1, T+1), (N, 1)).T #creates a TxN matrix where each rows are the Ts

        # 3) draw decision‐ and transition‐uniforms
        u_d   = np.random.rand(T, N, 2)   # binary install draws for firm I/J
        u_tr  = np.random.rand(T, N)      # for sampling next‐state via flattened F

        # 4) preallocate arrays
        #    last axis in `state` is [n_i, n_j, FC]
        state = np.zeros((T, N, 3), dtype=int)
        action = np.zeros((T, N, 2), dtype=int)

        # 5) initialize at t=0
        state[0,:,0] = np.random.randint(par.n_vals, size=N)    # n_i(0)
        state[0,:,1] = np.random.randint(par.n_vals, size=N)    # n_j(0)
        state[0,:,2] = np.random.randint(par.FC_vals,  size=N)    # FC(0)

        # 6) simulate
        for it in range(T):
            n_i = state[it,:,0]
            n_j = state[it,:,1]
            FC  = state[it,:,2]

            # --- firm decisions ---
            #probability of investing 
            p_i = ccps_i[n_i, n_j, FC, :]
            p_j = ccps_j[n_j, n_i, FC, :]
            action[it,:,0] = (u_d[it,:,0] < p_i[:,1]).astype(int)   # If the uniform draw is less than probability of investing, action = 1 
            action[it,:,1] = (u_d[it,:,1] < p_j[:,1]).astype(int)   # Same for player 2 

            # --- joint transition using your 3D kernel f_x ---
            for k in range(N):
                Fmat = transitions.f_x(par,
                           n_i[k], n_j[k], FC[k],
                           p_i[k], p_j[k])
                # Fmat has shape (n_i_states, n_j_states, FC_states)
                # and Fmat[a,b,c] = Pr(n_i'=a, n_j'=b, FC'=c | current state & actions)

                # Flatten that 3D array to a 1D vector so we can do discrete sampling 
                flat = Fmat.ravel()
                # Build the cumulative distribution (CDF) over that flattened vector
                cum  = flat.cumsum()
                # Draw a state 
                draw = u_tr[it,k] 

                # Find which index that draw falls into
                idx0 = cum.searchsorted(draw)

                # Inverse-flatten: this function takes the index of a 1D vector and the shape of the original matrix 
                ni1, nj1, fc1 = np.unravel_index(idx0, Fmat.shape) #returns the tuple of 3D indices that locate that element in the 3D matrix


                if it+1 < T:
                    state[it+1,k,0] = ni1
                    state[it+1,k,1] = nj1
                    state[it+1,k,2] = fc1

        # 7) flatten to DataFrame
        rows = T * N
        df = pd.DataFrame({
            'id':   idx.reshape(rows, order='F'),
            't':    t.reshape(rows, order='F'),

            'n_i':  state[:,:,0].reshape(rows, order='F'),
            'n_j':  state[:,:,1].reshape(rows, order='F'),
            'FC':   state[:,:,2].reshape(rows, order='F'),

            'a_i':  action[:,:,0].reshape(rows, order='F'),
            'a_j':  action[:,:,1].reshape(rows, order='F'),
        })

        return df



  