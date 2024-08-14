import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import time
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Gauss
from param import *

#np.seterr(all='raise')

T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times
 

# Keep the distribution of arms consistent each run
seed=0
arm_start,param_start,chg_dist=generate_arm_Gauss(T,K,m,seed)

# arm_start, param_start =['G', 'G', 'G'], [[0.9,0.5], [0.5,0.5], [0.4,0.5]]
# chg_dist = {'2500': [['G', 'G', 'G'], [[0.4,0.5], [0.8,0.5], [0.5,0.5]]],
#             '4500': [['G', 'G', 'G'], [[0.3,0.5], [0.2,0.5], [0.7,0.5]]],
#             '7000': [['G', 'G', 'G'], [[0.9,0.5], [0.8,0.5], [0.4,0.5] ]]
#            }
t1=time.time()
mab = GMAB(arm_start, param_start, chg_dist)

SW_UCB_data = mab.MC_regret('SW_UCB', N, T, param_swucb, store_step=1)
DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=1)
SW_TS_data = mab.MC_regret('SW_TS_gaussian', N, T, param_swts,store_step=1)
#SW_TS_data_sta = mab.MC_regret('SW_TS_gaussian_sta', N, T, param_swts,store_step=1)
DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=1)
#DS_TS_data = mab.MC_regret('D_TS_gaussian', N, T, param_dsts_g,store_step=1)
TS_data = mab.MC_regret('TS_gaussian', N, T,{},store_step=1)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=1)

rr=np.zeros(8)
L=['DS_UCB_data','SW_UCB_data','SW_TS_data','DS_TS_data','LBSDA_data','TS_data']
ii=0
for i in L:
    print(i+":",eval(i)[0][-1])
    rr[ii]=eval(i)[0][-1]
    ii += 1

print("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
t2=time.time()
print("time: ",t2-t1)

x=np.arange(T)
d = int(T / 20)
dd=int(T/1000)
xx = np.arange(0, T, d)
xxx=np.arange(0,T,dd)
alpha=0.05
plt.figure(2)

LBSDA_data1=LBSDA_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(LBSDA_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='SW-LB-SDA')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='c')

DS_UCB_data1=DS_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_UCB_data[0][xx], '-bd', markerfacecolor='none', label='DS-UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='b')

SW_UCB_data1=SW_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(SW_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, SW_UCB_data[0][xx], '-g^', markerfacecolor='none', label='SW-UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='g')

TS_data1=TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, TS_data[0][xx], color='brown',marker='*', markerfacecolor='none', label='TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='brown')

SW_TS_data1=SW_TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(SW_TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, SW_TS_data[0][xx], '-y^', markerfacecolor='none', label='SW-TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='y')

DS_TS_data1=DS_TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_TS_data[0][xx], '-ro', markerfacecolor='none', label='DS-TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='r')

plt.legend()
plt.xlabel('Round t')
plt.ylabel('Regret')
plt.savefig('pics/gauss2.pdf')
plt.show()
