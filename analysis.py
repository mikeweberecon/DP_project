
import pandas as pd
import aux 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#############
#DATA FRAMES#
#############
def data_eq_Mini(par, x_star):
    """
    Unpack the equilibrium CCP vector x_star into two 4-D arrays
    and tabulate only the probability of entry (action=1) for each player.
    If filter_playable is True, returns only the rows for (n_i=0, n_j=0).
    """
    # 1) unpack into 4-D CCPs for players i & j
    #    shape(ccps_i) = (n_i_vals, n_j_vals, FC_vals, n_actions)

    # turn whatever x_star is into a flat 1-D array
    ccps_i, ccps_j = aux.prep_input(par, x_star)

    # 2) build a table of (n_i, n_j, FC, p_i_entry, p_j_entry)
    rows = []
    for ni in range(par.n_vals):
        for nj in range(par.n_vals):
            for fci in range(par.FC_vals):
                rows.append({
                    'n_i':         ni,
                    'n_j':         nj,
                    'FC':          par.FC_grid[fci],
                    'p_i_entry':   ccps_i[ni, nj, fci, 1],
                    'p_j_entry':   ccps_j[ni, nj, fci, 1],
                })

    df = pd.DataFrame(rows)
    df = df[(df['n_i'] == 0) & (df['n_j'] == 0)]
    df = (
    df
    .query("n_i == 0 and n_j == 0")
    .loc[:, ["n_i", "n_j", "FC", "p_i_entry", "p_j_entry"]]
)
    # 3) print as markdown
    df.to_csv("entry_probs_mini.csv", index=False)
    return df


def data_eq_Main(par, x_star):
    # 1) unpack into 4-D CCPs
    ccps_i, ccps_j = aux.prep_input(par, x_star)

    # 2) collect only those (n_i, n_j) ≤ max_n
    rows = []
    for ni in range(par.n_vals):
        for nj in range(par.n_vals):
                for fci, fc in enumerate(par.FC_grid):
                    rows.append({
                        'n_i':        ni,
                        'n_j':        nj,
                        'FC':         fc,
                        'p_i_entry':  ccps_i[ni, nj, fci,1],
                        'p_j_entry':  ccps_j[ni, nj, fci,1],
                    })

    # 3) build the DataFrame
    df = pd.DataFrame(rows)
    df = df[df['n_i'] + df['n_j'] < par.n_bar]

    # 4) print and return
    df.to_csv("entry_probs_main.csv", index=False)
    return df


def data_eq_Extra(par, x_star):
    """
    Build a DataFrame of equilibrium CCPs when actions ∈ {0,…,5}.
    Rows are (n_i, n_j, FC, action, p_i, p_j).
    Optionally only keep n_i+n_j < par.n_bar.
    """
    # 1) unpack into 4-D CCPs: shape (n_i, n_j, FC, n_actions)
    ccps_i, ccps_j = aux.prep_input(par, x_star)

    rows = []
    for ni in range(par.n_vals):
        for nj in range(par.n_vals):
            # drop infeasible aggregates:
            if ni + nj >= par.n_bar:
                continue

            for fci in range(par.FC_vals):
                for action in range(par.k):   # k_vals = 6 here
                    rows.append({
                        'n_i':      ni,
                        'n_j':      nj,
                        'FC':       par.FC_grid[fci],
                        'action':   action,
                        'p_i':      ccps_i[ni, nj, fci, action],
                        'p_j':      ccps_j[ni, nj, fci, action],
                    })

    df = pd.DataFrame(rows)
    df.to_csv("action_probs_extra.csv", index=False)
    return df





#############
   #PLOTS#
#############

def plot_eq_Mini(par, x_star):
    """
    Plot the equilibrium CCPs for player i at state (n_i=0, n_j=0)
    across all FC levels.
    """
    # Unpack the CCP arrays from your equilibrium vector
    ccps_i, ccps_j = aux.prep_input(par, x_star)
    
    # Grab the FC grid and the CCP slice at (0,0)
    fc_values = par.FC_grid          
    p_entry   = ccps_i[0, 0, :, 1]       
    
    # Make the plot
    plt.figure()
    plt.plot(fc_values, p_entry, marker='o')
    plt.xlabel('Fixed Cost (FC)')
    plt.ylabel('Entry Probability at (n_i=0, n_j=0)')
    plt.title('Equilibrium CCPs: Probability of Entry vs. FC')
    plt.tight_layout()
    plt.show()

def plot_eq_Main(par, x_star, n_i=0):
    """
    3D surface of player i entry CCP at fixed n_i over (n_j, FC).
    """
    # unpack your ccps
    ccps_i, _ = aux.prep_input(par, x_star)       # ccps_i.shape == (n_vals, n_vals, FC_vals, action_choices)
    # we only want the probability of action=1:
    p_entry = ccps_i[..., 1]                      # shape (n_i, n_j, FC_vals)

    # slice out the plane n_i = fixed
    Z = p_entry[n_i, :, :]                        # shape (n_j, FC_vals)

    # get axis grids
    n_j_vals = par.n_grid                         # length n_vals
    FC_vals = par.FC_grid                         # length FC_vals
    N_j, FC = np.meshgrid(n_j_vals, FC_vals, indexing='ij')

    # plot
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(N_j, FC, Z,
                           rstride=1, cstride=1,
                           cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('n_j')
    ax.set_ylabel('Fixed FC')
    ax.set_zlabel('Pr(entry) of i')
    ax.set_title(f'Entry CCP (n_i={n_i})')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Pr(entry)')
    plt.tight_layout()
    plt.show()

def plot_nj_fc_surface(par, x_star, action=1, n_i=0, cmap='viridis'):
    """
    3D surface of the equilibrium CCP for player i choosing `action`,
    as n_j runs 0..n_vals-1 and FC runs over par.FC_grid, holding n_i fixed.
    """


    # unpack the full CCP arrays (shape: [n_i, n_j, FC, n_actions])
    ccps_i, _ = aux.prep_input(par, x_star)

    # build grids
    n_js    = np.arange(par.n_vals)
    FCs     = par.FC_grid
    Nj, FCg = np.meshgrid(n_js, FCs, indexing='ij') 
    # slice out the plane at n_i and action
    P       = ccps_i[n_i, :, :, action]          # shape (n_j, FC)

    # make the 3D plot
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Nj, FCg, P, cmap=cmap, edgecolor='k', linewidth=0.2)
    ax.set_xlabel(r'$n_j$')
    ax.set_ylabel('Fixed Cost (FC)')
    ax.set_zlabel(f'$P_i(a={action}\mid n_i={n_i})$')
    ax.set_title(f'Equilibrium CCP, action={action}, n_i={n_i}')
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Probability')
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_action_fc_surface(par, x_star, n_i=0, n_j=0, cmap='viridis'):
    """
    3D surface of action probabilities P_i(a | n_i, n_j, FC)
    as action and FC vary, holding n_i and n_j fixed.
    """
   
    # Unpack into CCPs array of shape (n_i, n_j, FC, n_actions)
    ccps_i, _ = aux.prep_input(par, x_star)

    # Slice to get an (FC_vals × n_actions) array
    probs = ccps_i[n_i, n_j, :, :]

    

    # Build meshgrid: actions on X, FC on Y
    n_actions = probs.shape[1]
    actions = np.arange(n_actions)
    FCs = par.FC_grid
    A, F = np.meshgrid(actions, FCs, indexing='xy')
    Z = probs  

    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, F, Z, cmap=cmap, edgecolor='k', linewidth=0.3, alpha=0.9)
    ax.set_xlabel('Action $a$')
    ax.set_ylabel('Fixed Cost (FC)')
    ax.set_zlabel(r'$P_i(a\mid n_i=%d,n_j=%d)$' % (n_i, n_j))
    ax.set_title(f'Equilibrium Action Probabilities at $n_i={n_i}, n_j={n_j}$')
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Probability')
    plt.tight_layout()
    plt.show()

    return fig, ax


def get_ccps(model):
    """
    Unpack conditional choice probabilities for each firm from a solved model.
    Returns (ccps_i, ccps_j, par).
    """
    par = model.par
    # Flatten the solution vector
    x_vec = model.sol.x.ravel()
    # Unpack into two CCP arrays
    ccps_i, ccps_j = aux.prep_input(par, x_vec)
    return ccps_i, ccps_j, par




def plot_main_equilibrium(model, nj_fixed=0):
    """
    For binary_choice_main: plot CCP_i vs n_i for fixed rival state n_j,
    across all FC levels.
    """
    ccps_i, _, par = get_ccps(model)
    fig, ax = plt.subplots()
    x = np.arange(par.n_vals)
    for f in range(par.FC_vals):
        y = ccps_i[:, nj_fixed, f, 1]
        ax.plot(x, y, label=f'FC idx {f}')
    ax.set_title(f'Main: Pr(install) vs n_i | n_j={nj_fixed}')
    ax.set_xlabel('n_i')
    ax.set_ylabel('Pr(install)')
    ax.legend()
    plt.show()


def plot_multiple_equilibrium(model, fc_idx=0, choice_idx=0):
    """
    For MultipleChoice: 3D surface of CCP for a given choice over (n_i, n_j)
    at one FC slice.
    """
    ccps, _, par = get_ccps(model)
    X, Y = np.meshgrid(np.arange(par.n_vals), np.arange(par.n_vals))
    Z = ccps[:, :, fc_idx, choice_idx].T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none')
    ax.set_title(f'MultipleChoice: Pr(choice={choice_idx}) at FC idx {fc_idx}')
    ax.set_xlabel('n_i')
    ax.set_ylabel('n_j')
    ax.set_zlabel('Probability')
    plt.show()