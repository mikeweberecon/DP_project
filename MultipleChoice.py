#%%
import numpy as np
import scipy.optimize as opt
import time
import aux

import solvers
from types import SimpleNamespace


class multiple_choice:
    def __init__(self):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.setup()

    def setup(self):
        par = self.par
        sol = self.sol

        # --- model parameters ---
        par.n_bar   = 5
        par.p      = 11
        par.beta   = 0.99
        par.sigma  = 1.0
        par.Nplayers = 2

        # fixed‐cost tech
        par.min_FC = 0
        par.max_FC = 5
        par.p_FC   = 0.45

        # action space
        par.k = 3
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
        sol.iterations = 1
        sol.dec        = 4
        # raw_sol: store up to `iterations` distinct equilibria
        N_P = aux.input_vector_indx(par)
        sol.raw_sol = np.zeros((sol.iterations, 2*N_P)) #shape is a 2D matrix, rows are iterations, and columns are ccps of i, and then ccps of j in one vector given each combination of states.
        sol.raw_sol_polit = np.zeros((sol.iterations, 2*N_P))


    def solve(self,method="root"):
        sol = self.sol
        if method == "root":
            self.sol.x = solvers.root_solver(self.par,sol)
        elif method == "policy":
            self.sol.x = solvers.policy_iteration_solver(self.par,sol)
        print("Solution:", self.sol.x)
        return self.sol.x



  