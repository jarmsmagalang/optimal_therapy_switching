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

            #tau  NT
tauNTvals = [[10, 2], #Fig. 3a1
             [1,  2], #Fig. 3a2
             [3,  1], #Fig. 3a3
             [6,  6]] #Fig. 3a4

#Set figure titles
titles = ["Fig. 3a1", "Fig. 3a2", "Fig. 3a3", "Fig. 3a4"]

for j in range(len(tauNTvals)):
    #Through all possible values of tau and N_T
    
    tau = tauNTvals[j][0] #Switching rate
    NT = tauNTvals[j][1] #Number of simultaneous therapies
    
    WR=(1/tau)*(1/365) #Take inverse of switching rate
    
    paramvals=[[D,v,WR,NT,R1,R0], #With switching
               [D,v,0 ,NT,R1,R0]] #Without switching
    
    colors = ["green", "red"]
    fig1, ax1 = plt.subplots()
        
    for i in range(len(paramvals)):
        params = paramvals[i]
        
        #Compute coupled continuous RDT over 1000 simulations
        sim_rdtvals = [[coupled.cont_sim_RDT(params,r0) for _ in range(1000)] for r0 in eta0vals]
        
        #Compute for mean RDT
        sim_mrdt = [np.mean(sim_rdtvals[i]) for i in range(len(sim_rdtvals))]
        
        #Compute for standard error of mean RDT
        sim_mrdt_error = [np.std(sim_rdtvals[i])/np.sqrt(1000) for i in range(len(sim_rdtvals))]
        
        #Compute ana. mean RDT, taken from Eq. 6
        ana_mrdt = [coupled.cont_ana_MRDT(params,r0) for r0 in eta0vals]
        
        #Begin plotting
        ax1.plot(eta0vals, [i/365 for i in ana_mrdt], color = colors[i])
        ax1.errorbar(eta0vals, [i/365 for i in sim_mrdt], yerr = [i/365 for i in sim_mrdt_error], fmt = "o", capsize = 5, color = colors[i])
    
    #Set axis label and figure titles
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\langle T \vert \eta \rangle$ (yr)")
    ax1.set_title(titles[j])