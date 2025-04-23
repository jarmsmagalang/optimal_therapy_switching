import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.H_fixedpt import H_fixedpt
import matplotlib

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams['axes.linewidth'] = 3

#Initialize N_T = 2 possible therapies
eta1vals = np.linspace(0,1,101)
eta2vals = np.linspace(0,1,101)

H0vals = pd.DataFrame(columns = eta1vals, index= eta2vals)

for j in range(len(eta1vals)):
    for i in range(len(eta2vals)):
        
        #For each eta1 and eta2, compute for the fixed point in H
        H0vals.iloc[i,j] = H_fixedpt(eta1vals[i], eta2vals[j])
H0vals = H0vals.apply(pd.to_numeric, errors='coerce')

#Normalize fixed points by the maximum possible value
H0max = H0vals.values.max()
H0vals = H0vals/H0max
H0vals = H0vals[::-1]


fig1, ax1 = plt.subplots()
#Plot the contour map of the fixed points
eta_x, eta_y = np.meshgrid(H0vals.index.to_numpy(), H0vals.columns.to_numpy())
cs = ax1.contourf(eta_x, eta_y, H0vals.values.T, cmap = "rocket")

#Set axis labels
ax1.set_xlabel("$\eta_{1;0}$", fontsize = 30)
ax1.set_ylabel("$\eta_{2;0}$", fontsize = 30)

#Set colorbar
cbar = fig1.colorbar(cs)
cbar.set_label('$\\tilde{H}/H_0$', fontsize = 30)
cbar.ax.tick_params(labelsize=20) 

#Set tick labels
ax1.tick_params(axis="x", labelsize=30)
ax1.tick_params(axis="y", labelsize=30)
ax1.set_xticks(np.linspace(0,1,3))
ax1.set_yticks(np.linspace(0,1,3))