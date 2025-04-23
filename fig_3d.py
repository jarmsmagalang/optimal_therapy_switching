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

tauNTvals = [[ [tau, 2] for tau in np.arange(2,11,2) ], #Constant NT, varying tau
             [ [3, NT] for NT in np.arange(2,11,2) ]]   #Constant tau, varying NT

#Set labels and figure titles
titles = ["Fig. 3d1", "Fig. 3d2"]
labels = [r"$\tau$ (yr)", r"$N_T$"]
ticks = np.arange(2,11,2)
for j in range(len(tauNTvals)):
    
    spvals = []
    smvals = []
    for k in range(len(tauNTvals[j])):
        tau = tauNTvals[j][k][0] #Switching rate
        NT = tauNTvals[j][k][1] #Number of simultaneous therapies
        
        WR=(1/tau)*(1/365) #Take inverse of switching rate
        
        paramvals=[[D,v,WR,NT,R1,R0], #With switching
                   [D,v,0 ,NT,R1,R0]] #Without switching
        
        colors = ["green", "red"]
        
        ana_mrdtvals = []
        
        for i in range(len(paramvals)):
            params = paramvals[i]
            
            #Compute ana. mean RDT, taken from Eq. 6, for each possible initial position
            ana_mrdt = [coupled.cont_ana_MRDT(params,r0) for r0 in eta0vals]
            ana_mrdtvals.append(np.array(ana_mrdt))
        
        #Compute difference of MRDTs with and without therapy switching
        ana_mrdt_diff = ana_mrdtvals[0]-ana_mrdtvals[1]
        
        #Find index of where the sign in ana_mrdt_diff changes
        eta_th_idx = [i for i,x in enumerate((np.diff(np.sign(np.real(ana_mrdt_diff))) != 0)) if x][1]
        
        #Obtain S+ and S- as defined in Fig. 3c
        splus = np.trapz(np.abs(ana_mrdt_diff[eta_th_idx:-1]), eta0vals[eta_th_idx:-1])
        sminus = np.trapz(np.abs(ana_mrdt_diff[0:eta_th_idx]), eta0vals[0:eta_th_idx])
        spvals.append(splus)
        smvals.append(sminus)
    
    #Begin plotting bar plot of S+ and S-
    fig1, ax1 = plt.subplots()   
    ax1.bar(np.arange(len(ticks)), spvals, 0.3, label = r"$S_+$")
    ax1.bar(np.arange(len(ticks))+0.3, smvals, 0.3, label = r"$S_+$")
    ax1.set_xticks(np.arange(len(ticks)) + 0.3 / 2, ticks)
    
    #Set tick labels and titles
    ax1.set_xlabel(labels[j])
    ax1.set_ylabel(r"$S$")
    ax1.set_title(titles[j])
    ax1.set_yscale("log")
    ax1.legend()