#%%
import numpy as np
import scipy.optimize as opt
import time
import aux
import operators
import pandas as pd
from types import SimpleNamespace

def objective_function(x, par):
    # unpack CCP guesses
    ccps_one_pre, ccps_two_pre = aux.prep_input(par, x) # x is a flat vector containing all ccps for both players. By running this function it converts it to 2 4D shape array. (ccps_one_pre, ccps_two_pre)

    # 1) Value‐function step
    EV_one_post = operators.bellman_operator_14(par, ccps_one_pre, ccps_two_pre)    #solve the reduced bellman operator for each player given this 4D shape of ccps
    EV_two_post = operators.bellman_operator_14(par, ccps_two_pre, ccps_one_pre)    

    # 2) CCP best‐response
    ccps_one_post = operators.compute_ccps_i(par, ccps_one_pre, ccps_two_pre, EV_one_post) #compute updated ccps given these value functions (Best response)
    ccps_two_post = operators.compute_ccps_i(par, ccps_two_pre, ccps_one_pre, EV_two_post)

    # 3) pack & return residual
    x_out = aux.prep_output(par, ccps_one_post, ccps_two_post)   #takes the 4D array result and runs prep_output to flatten everything into 1 vector of ccps of both players. 
    return x_out - x     #returns the difference between the updated vector and the old vector (x) before performing this mapping. 

class dynamic_discrete_model:
    def __init__(self):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.setup()

    def setup(self):
        par = self.par
        sol = self.sol

        # --- model parameters ---
        par.n_bar   = 11
        par.p      = 1.0
        par.beta   = 0.95
        par.sigma  = 1.0
        par.Nplayers = 2

        # fixed‐cost tech
        par.min_FC = 1
        par.max_FC = 10
        par.p_FC   = 0.5

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
        par.reps = 100
        

        # solver settings
        sol.iterations = 100
        sol.dec        = 4
        # raw_sol: store up to `iterations` distinct equilibria
        N_P = aux.input_vector_indx(par)
        sol.raw_sol = np.zeros((sol.iterations, 2*N_P)) #shape is a 2D matrix, rows are iterations, and columns are ccps of i, and then ccps of j in one vector given each combination of states.
        sol.raw_sol_polit = np.zeros((sol.iterations, 2*N_P))


    def solve(self):
        par = self.par
        sol = self.sol

        eq_counter = 0
        t0 = time.perf_counter()

        for it in range(sol.iterations):
            print(f"Iteration {it+1}/{sol.iterations}")

            guess  = aux.draw_guess(par)   #draws a random uniform guess of ccps into a 1D vector where ccps_i and ccps_j are concatenated. 
            result = opt.root(objective_function,
                              guess,
                              args=(par,),
                              method='hybr',
                              options={'xtol':1e-12,'maxfev':10000}) #finds the fixed point in the operator P = BR(P)

            if not result.success:
                print("root-finder failed:", result.message)
                continue

            x_new = result.x
            is_new = aux.check_new_eq(sol, x_new)
            if is_new:
                sol.raw_sol[eq_counter] = x_new    #collects solution into the 2D array "raw_sol"
                eq_counter += 1

            aux.print_eq(is_new, eq_counter)

        t1 = time.perf_counter()
        sol.time_elapsed = t1 - t0
        sol.eq_counter  = eq_counter
        # store only the distinct equilibria found
        sol.sol = sol.raw_sol[:eq_counter]

        print(f"\nDone in {sol.time_elapsed:.2f}s: found {eq_counter} equilibrium(s).")

    def solve_pol_iter(self,
                   tol     = 1e-8,   # stopping rule ‖P^{new}–P^{old}‖_∞
                   max_it  = 20,    # fail-safe
                   relax   = 0.5):  # dampening; 1.0 = pure policy-iteration
        par, sol = self.par, self.sol

    # Pre‐allocate space for up to par.reps distinct equilibria
        N_P = aux.input_vector_indx(par)
        sol.raw_sol_polit = np.zeros((par.reps, 2*N_P))
        eq_counter = 0

        t0 = time.perf_counter()
        for rep in range(par.reps):
            print(f"=== Starting rep {rep+1}/{par.reps} ===")

            # 0) draw a fresh random CCP guess
            flat_guess = aux.draw_guess(par)           # 1D array length 2*N_P
            P1, P2     = aux.prep_input(par, flat_guess)  # 4-D arrays

            # 1) inner policy‐iteration loop
            for it in range(1, max_it+1):
                # 1a) value‐function step
                EV1 = operators.bellman_operator_14(par, P1, P2)
                EV2 = operators.bellman_operator_14(par, P2, P1)

                # 1b) best‐response step
                P1_new = operators.compute_ccps_i(par, P1, P2, EV1)
                P2_new = operators.compute_ccps_i(par, P2, P1, EV2)

                # 1c) check convergence
                diff1 = np.max(np.abs(P1_new - P1))
                diff2 = np.max(np.abs(P2_new - P2))
                diff  = max(diff1, diff2)

                if diff < tol:
                    print(f"  Converged in {it} iters (Δ = {diff:.2e})")
                    break

                # optional progress print
                if it % 5 == 0:
                    print(f"  it={it:>3d}: Δ1={diff1:.2e}, Δ2={diff2:.2e}")

                # advance
                P1, P2 = P1_new, P2_new
            else:
                raise RuntimeError(f"Policy‐iteration failed to converge in {max_it} iters "
                                f"(last Δ1={diff1:.2e}, Δ2={diff2:.2e})")

            # 2) pack the converged CCPs into a flat vector
            x_new = aux.prep_output(par, P1, P2)  # shape (2*N_P,)

            # 3) test for uniqueness and store if new
            is_new = aux.check_new_eq(sol, x_new)
            if is_new:
                sol.raw_sol_polit[eq_counter, :] = x_new
                eq_counter += 1

            aux.print_eq(is_new, eq_counter)

        # after all reps, slice off only the found equilibria
        sol.eq_counter     = eq_counter
        sol.sol_poli_iter  = sol.raw_sol_polit[:eq_counter, :]
        sol.time_elapsed   = time.perf_counter() - t0

        print(f"\nDone in {sol.time_elapsed:.2f}s: "
            f"found {eq_counter} equilibrium(s).")
  

# usage
dm = dynamic_discrete_model()
dm.solve_pol_iter()



# ------  pull objects from the solved model  ------------------
par        = dm.par
sol_array  = dm.sol.sol_poli_iter        # shape (n_eq, 32)   (may be empty)

# quick exit if no equilibria were found
if sol_array.size == 0:
    print("No equilibria stored in dg.sol.sol")
else:
    n_eq   = sol_array.shape[0]

    # dimensions
    k          = par.k          # 2 actions
    n_i_dim    = par.n_vals         # 2 values of n_i
    n_j_dim    = par.n_vals         # 2 values of n_j
    fc_dim     = par.FC_vals         # 2 FC indices (0,1)
    cells_per_player = k * n_i_dim * n_j_dim * fc_dim   # = 16

    rows = []

    # ------  loop over each equilibrium vector  ----------------
    for eq_id in range(n_eq):

        flat = sol_array[eq_id]

        # split into player‑1 block and player‑2 block
        ccp1_flat = flat[:cells_per_player]
        ccp2_flat = flat[cells_per_player:]

        # reshape to (n_i, n_j, fc, action)
        ccp1 = ccp1_flat.reshape(n_i_dim, n_j_dim, fc_dim, k)
        ccp2 = ccp2_flat.reshape(n_i_dim, n_j_dim, fc_dim, k)

        # loop over state and action
        for n_i in range(n_i_dim):
            for n_j in range(n_j_dim):
                for fc_idx in range(fc_dim):
                    FC_val = par.FC_grid[fc_idx]   # actual FC level (2 or 3)
                    for a in range(k):

                        rows.append({
                            "equilibrium": eq_id,
                            "n_i":        n_i,
                            "n_j":        n_j,
                            "FC":         FC_val,
                            "player":     1,
                            "action":     a,
                            "probability": ccp1[n_i, n_j, fc_idx, a]
                        })

                        rows.append({
                            "equilibrium": eq_id,
                            "n_i":        n_i,
                            "n_j":        n_j,
                            "FC":         FC_val,
                            "player":     2,
                            "action":     a,
                            "probability": ccp2[n_i, n_j, fc_idx, a]
                        })

    # ------  build tidy DataFrame  ------------------------------
    df = pd.DataFrame(rows)


    print(df)
# %%
n = 10*2*11
pd.set_option('display.max_rows', None)      # or some large int instead of None
pd.set_option('display.max_columns', None)

print(df)
# %%
