import numpy as np
import time
import aux
import objective_function as of
import scipy.optimize as opt


def policy_iteration_solver(par, sol, tol=1e-8, max_it=20):

    # 0) prepare storage
    N_P = aux.input_vector_indx(par)               # number of P entries per player
    sol.raw_sol = np.zeros((sol.reps, 2*N_P))      # space for up to par.reps equilibria
    eq_counter = 0

    t0 = time.perf_counter()

    # 1) loop over random starts
    for rep in range(sol.reps):
        print(f"=== Starting rep {rep+1}/{sol.reps} ===")
        old_P = aux.draw_guess(par)  # flat length-2N_P vector

        # 2) inner “policy‐iteration” via objective_function
        for it in range(1, max_it+1):
            
            #Call objective function
            new_P = of.objective_function(old_P, par)
            #Check convergence 
            diff  = np.max(np.abs(new_P - old_P))

            if diff < tol:
                print(f"Converged after {it} iterations (Δ = {diff:.2e})")
                
                # 3) check & store new equilibrium
                if aux.check_new_eq(sol, new_P):
                    sol.raw_sol[eq_counter, :] = new_P
                    eq_counter += 1
                    aux.print_eq(True, eq_counter)
                else: 
                    aux.print_eq(False,eq_counter)

                break   #  exit the it‐loop, proceed to next rep

            # update for next iter
            old_P = new_P

            if it % 5 == 0:
                print(f"Iteration {it}: Δ = {diff:.2e}")
        else:
            # only runs if the for‐loop didn’t break
            raise RuntimeError(f"Policy iteration failed to converge in {max_it} iters (last Δ={diff:.2e})")

    # 4) after all reps, finalize sol
    sol.eq_counter   = eq_counter
    sol.sol          = sol.raw_sol[:eq_counter, :]
    sol.time_elapsed = time.perf_counter() - t0

    print(f"\nDone in {sol.time_elapsed:.2f}s: found {eq_counter} equilibrium(s).")
    return sol.sol

    


def root_solver(par,sol):

    eq_counter = 0

    t0 = time.perf_counter()
    
    for it in range(sol.iterations):

        print(f'Iteration {it+1} of {sol.iterations}:')

    # Draw guess
        guess  = aux.draw_guess(par)
    
    # Call root finder
        result = opt.root(of.objective_for_root, guess, args=(par,), method='hybr', options={'xtol':1e-12, 'maxfev':10000})

        new_sol_conj = result.x

    # check if equilibrium is known
        new_eq_bool = aux.check_new_eq(sol, new_sol_conj)

        if new_eq_bool: # new eq found

            sol.raw_sol[eq_counter] = new_sol_conj # store solution by indexing on eq counter

            eq_counter += 1 # update counter
    
        aux.print_eq(new_eq_bool, eq_counter) # print

    t1 = time.perf_counter()

    sol.time_elapsed = t1 - t0

    print(f'\n{sol.iterations} iterations took {sol.time_elapsed:.2f} seconds')

     
        # store number of eq
    sol.eq_counter = eq_counter
    
        # save only polished eq
    sol.sol = sol.raw_sol[:eq_counter]

    if not result.success:
            print("Root-finder failed:", result.message)

    return sol.sol
       
