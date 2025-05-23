#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinaryChoiceMini import binary_choice_mini
from BinaryChoiceMain import binary_choice_main

def simple_sim(game, N = 500, T = 200, seed = 2020):
    # 1a) simulate
    df = game.sim_data( N=N, T=T,seed = seed)   # returns columns: id, t, n_i, n_j, FC, a_i, a_j, slot_i, slot_j (if you track slots)

    # 1b) compute cumulative installs per firm
    df['cum_i'] = df.groupby('id')['a_i'].cumsum()
    df['cum_j'] = df.groupby('id')['a_j'].cumsum()

    # 1c) summary stats by period
    summary = (df
        .groupby('t')
        .agg(mean_i = ('cum_i','mean'),
            p05_i  = ('cum_i', lambda x: np.percentile(x,5)),
            p95_i  = ('cum_i', lambda x: np.percentile(x,95)),
            mean_j = ('cum_j','mean'),
            p05_j  = ('cum_j', lambda x: np.percentile(x,5)),
            p95_j  = ('cum_j', lambda x: np.percentile(x,95)))
        .reset_index()
    )

    # 1d) plot mean ± 5–95 bands
    fig, ax = plt.subplots()
    ax.fill_between(summary.t, summary.p05_i, summary.p95_i)
    ax.plot(summary.t, summary.mean_i)
    ax.fill_between(summary.t, summary.p05_j, summary.p95_j)
    ax.plot(summary.t, summary.mean_j)
    ax.set_xlabel('Period t')
    ax.set_ylabel('Cumulative installs')
    ax.set_title('Baseline Monte Carlo paths')
    fig.show()

    return fig, ax

def install_time_cdf(game, N=2000, T=500, seed=2025):
    """
    Simulate the game N times for T periods and plot the CDF
    of the *first*-install time for each firm.
    Returns (fig, ax).
    """
    # 1) simulate panel
    df = game.sim_data(N=N, T=T, seed=seed)

    # 2) for each run, find the period of the first install (a_i==1)
    first_i = df[df['a_i']==1].groupby('id')['t'].min()
    first_j = df[df['a_j']==1].groupby('id')['t'].min()

    # if some never install, set their time to T+1 so they never count
    first_i = first_i.reindex(range(1, N+1), fill_value=T+1)
    first_j = first_j.reindex(range(1, N+1), fill_value=T+1)

    # 3) build empirical CDFs
    x = np.arange(1, T+1)
    cdf_i = [(first_i <= tt).mean() for tt in x]
    cdf_j = [(first_j <= tt).mean() for tt in x]

    # 4) plot
    fig, ax = plt.subplots()
    ax.plot(x, cdf_i, label='Firm I')
    ax.plot(x, cdf_j, label='Firm J')
    ax.set_xlabel('Period $t$')
    ax.set_ylabel(r'$\Pr(\text{first install}\leq t)$')    
    ax.set_title('Installation‐time CDF')
    ax.legend()

    return fig, ax

