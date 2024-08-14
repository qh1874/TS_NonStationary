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
reward_u_p = 1
sigma_max = 1/2

gamma_D_UCB = 1 - 1/(4*reward_u_p)*np.sqrt(Gamma_T_garivier/T)
tau_theorique =np.sqrt(T*np.log(T)/Gamma_T_garivier)

#CUMSUM
h_CUSUM = np.log(T/Gamma_T_garivier)
alpha_CUSUM = np.sqrt(Gamma_T_garivier/T*h_CUSUM)
M_CUSUM = 50
eps_CUSUM = 0.05

#M-UCB
w_BRANO = 800
b_BRANO = np.sqrt(w_BRANO/2*np.log(2*K*T**2))
gamma_MUCB = np.sqrt(Gamma_T_garivier*np.log(T)*K/T)
delta_min = 0.2

param_dsucb={'B':1,'ksi':2/3, 'gamma': gamma_D_UCB}
param_swucb={'C': 1, 'tau': int(2*tau_theorique)}
param_swts={ 'sigma':sigma_max, 'tau': int(np.sqrt(T/nb_change)*np.log(T))}
param_dsts={'sigma':sigma_max,'gamma': 1-np.sqrt(nb_change/T/np.log(T))}
param_lbsda={'tau': int(2*tau_theorique)}
param_cumsum={'alpha':alpha_CUSUM , 'h': h_CUSUM, 'M':M_CUSUM, 'eps':eps_CUSUM, 'ksi':1/2}
param_mucb={'w':w_BRANO, 'b':b_BRANO, 'gamma':gamma_MUCB}
