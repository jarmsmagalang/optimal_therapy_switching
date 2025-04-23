import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import src.coupled_continuous as coupled

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams['axes.linewidth'] = 3

#Define parameters and functions for the coupled continuous models
D=10**(-4) #Diffusion constant
v=-8*10**(-5) #Drift constant
R1=1 #Reflecting circular boundary
R0=0.4 #Absorbing circular boundary
eta0vals = np.linspace(R0,R1,15) #Initial positions

all_paramvals = [[[D,v,(1/tau)*(1/365),2,R1,R0] for tau in [1,4,7,10]], #Varying tau, constant NT
                 [[D,v,(1/3)*(1/365),NT,R1,R0] for NT in [1,4,7,10]]]   #Varying NT, constant tau

#Set labels and figure titles
labels = [r"$\tau =$", r"$N_T =$"]
labels2 = [1,4,7,10]
titles = ["Fig. 3b1", "Fig. 3b2"]

for j in range(len(all_paramvals)):
    paramvals = all_paramvals[j]

    fig1, ax1 = plt.subplots()
        
    for i in range(len(paramvals)):
        params = paramvals[i]
        
        #Compute ana. mean RDT, taken from Eq. 6
        ana_mrdt = [coupled.cont_ana_MRDT(params,r0) for r0 in eta0vals]
        
        #Begin plotting
        ax1.plot(eta0vals, [i/365 for i in ana_mrdt], label = labels[j]+str(labels2[i]))
    
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\langle T \vert \eta \rangle$ (yr)")
    ax1.set_title(titles[j])
    if j == 1: #Remove log scale y axis for Fig. 3b2
        ax1.set_yscale("log")
    ax1.legend()