#%%
import numpy as np
from BinaryChoiceMini   import binary_choice_mini
from BinaryChoiceMain   import binary_choice_main
from MultipleChoice     import multiple_choice
import analysis
import simulation
import matplotlib.pyplot as plt

def run_model(model_class,method="policy"):
    model = model_class()
    model.setup()
    model.solve(method=method)
    par    = model.par
    x_star = model.sol.x
    x_star = x_star.ravel() 
    #data frame
    if model_class is binary_choice_mini:
        #data frame
        df_eq = analysis.data_eq_Mini(par,x_star)
        print(df_eq)
        #plot
        analysis.plot_eq_Mini(par, x_star)
        return model

    elif model_class is binary_choice_main: 
        df_eq = analysis.data_eq_Main(par,x_star)
        print(df_eq)
        analysis.plot_eq_Mini(par, x_star)
        analysis.plot_eq_Main(par,x_star,n_i=0)
        return model 
    
    elif model_class is multiple_choice:
        df_eq = analysis.data_eq_Extra(par,x_star)
        print(df_eq)
        analysis.plot_nj_fc_surface(par,x_star,action=1,n_i=0)
        analysis.plot_action_fc_surface(par,x_star,n_i =0,n_j = 0)
        analysis.plot_multiple_equilibrium
    

if __name__ == "__main__":
    model = run_model(binary_choice_main, method="root")
    
    #fig, ax = simulation.install_time_cdf(model, N=2000, T=50, seed=42)
    #plt.show()    



# %%
