import numpy as np

param={
    'T':100000, # round 
    'K':10,  # arm
    'm':10000, # length of stationary phase, breakpoints=T/m
    'N':10 # repeat times
}

T=param['T']
K=param['K']
m=param['m']
nb_change=int(T/m)
Gamma_T_garivier = int(T/m)
sigma_max = 5

gamma_D_UCB = 1 - 1/(4*(1+ 2*sigma_max))*np.sqrt(Gamma_T_garivier/T)
tau_theorique = 2*(1 + 2*sigma_max)*np.sqrt(T*np.log(T)/Gamma_T_garivier)

param_dsucb={'B':1,'ksi':2/3, 'gamma': gamma_D_UCB}
param_swucb={'C': 1, 'tau': int(tau_theorique)}
param_swts={ 'sigma':sigma_max, 'tau': int(sigma_max*np.sqrt(T/nb_change)*np.log(T))}
param_dsts={'sigma':sigma_max, 'gamma': 1-1/sigma_max*np.sqrt(nb_change/T/np.log(T))}
#param_lbsda={'tau': int(2*np.sqrt(T*np.log(T)/nb_change))}
param_lbsda={'tau': int(tau_theorique)}


