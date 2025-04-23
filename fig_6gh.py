import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import src.heterogeneous as heterogeneous

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size':30})
matplotlib.rcParams['axes.linewidth'] = 3

### Initialize parameters common to both limited and costed switching
runs = 10000
N = 30 #Number of states in the discrete uncoupled model
mu = 0.05 #Mutation probability
sigma = 0.5 #Mutation state transition probability
init_eta = 0.8 #Initial therapy efficacy

#All therapy switching rates considered to ensure a smooth curve in the figure
tauvals1 = np.append(np.concatenate([np.arange(10**(i), 10**(i+1), 10**(i)) for i in np.arange(-2,2, dtype = float) if i != 0.0]),100)
tauvals2 = np.array([i for i in np.arange(1,5,0.2) if i not in tauvals1])
tauvals3 = np.arange(5,10)
tauvals_all = np.concatenate((tauvals1, tauvals2, tauvals3))
tauvals = np.unique(tauvals_all)

#Equidistant (in log space) points to compute for the errors 
tauvals4 = np.array([i for i in np.array([10**i for i in np.arange(-2,2, 0.5)]) if i not in tauvals1])

#Begin computing mean RDTs for limited resetting
limvals = [5,10,15] #Considered values of $\ell$

#Initialize all parameters to be used
taulvals = [[tau,l, 0] for tau in tauvals for l in limvals]+[[tau, l, 1] for tau in tauvals4 for l in limvals]+[[np.inf, 0, 0]]
mrdtvals_lim = []
srdtvals_lim = []
tauplot_lim = []
for taul in taulvals:
    rdtvals = []
    
    r = (1/taul[0])*(1/365) #Inverse of therapy switching rate
    params = [N, mu, sigma, init_eta, r]

    #Simulate the runs for the uncoupled discrete model with limited switching
    for _ in range(runs):
        rdtvals.append(heterogeneous.heterogeneous_RDT(params, res_type = "limited", lim = taul[1]))
        
    mrdt = np.mean(rdtvals) #Mean
    srdt = np.std(rdtvals) #Standard deviation
    mrdtvals_lim.append(mrdt)
    srdtvals_lim.append(srdt)
    tauplot_lim.append(taul[2])

#Keep values in dataframe
df_lim = pd.DataFrame(data = {"tau": [i[0] for i in taulvals],
                              "l": [i[1] for i in taulvals],
                              "mrdt": mrdtvals_lim,
                              "srdt": srdtvals_lim,
                              "plot": tauplot_lim})

#Begin computing mean RDTs for costed resetting
costvals = [0.1,1,100] #Considered values of $c$

#Initialize all parameters to be used
taucvals = [[tau,c, 0] for tau in tauvals for c in costvals]+[[tau, c, 1] for tau in tauvals4 for c in costvals]+[[np.inf, np.inf, 0]]
mrdtvals_cst = []
srdtvals_cst = []
tauplot_cst = []
for tauc in taucvals:
    rdtvals = []
    
    r = (1/taul[0])*(1/365) #Inverse of therapy switching rate
    params = [N, mu, sigma, init_eta, r]

    for _ in range(runs):
        rdtvals.append(heterogeneous.heterogeneous_RDT(params, res_type = "costed", lim = tauc[1]))
        
    mrdt = np.mean(rdtvals) #Mean
    srdt = np.std(rdtvals) #Standard deviation
    mrdtvals_cst.append(mrdt)
    srdtvals_cst.append(srdt)
    tauplot_cst.append(tauc[2])

#Keep values in dataframe
df_cst = pd.DataFrame(data = {"tau": [i[0] for i in taucvals],
                              "c": [i[1] for i in taucvals],
                              "mrdt": mrdtvals_cst,
                              "srdt": srdtvals_cst,
                              "plot": tauplot_cst})

#Begin plotting
cmap = matplotlib.colormaps['coolwarm']

unique_l = df_lim.l.unique()
mfpt_l0 = df_lim[df_lim["l"] == 0]["mrdt"] #Case where there is no therapy switching

colors_l = cmap(np.linspace(0, 1, len(unique_l)))

fig1, ax1 = plt.subplots(figsize=(8.5,5))

#Plot for the case where there is no therapy switching/no allowed therapy swtiches
ax1.axhline(mfpt_l0.values[0]/365, lw = 3, color = colors_l[0], label = r"$\ell=0$")

for l in range(len(unique_l)):
    if unique_l[l] > 0:
        df_l = df_lim[df_lim["l"] == unique_l[l]].sort_values("tau")
        df_l2= df_l[df_l["plot"] == 1]
        ax1.errorbar(df_l2["tau"], df_l2["mrdt"]/365, fmt = "s", markersize = 7, capsize = 7, yerr=1.96*df_l2["mrdt"]/365, color = colors_l[l+1])
        ax1.plot(df_l["tau"], df_l["mrdt"]/365, label = r"$\ell=$"+str(unique_l[l]), color = colors_l[l+1], lw = 3)
        
ax1.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax1.tick_params(axis="y", labelsize=30, length = 10, width = 3)
ax1.set_xlabel(r"$\tau$ (yr)", fontsize = 30)
ax1.set_ylabel(r"$\langle T \vert \eta \rangle$ (yr)", fontsize = 30)
ax1.legend(frameon=False, fontsize = 25, handlelength=1, loc = "upper left")
ax1.set_xscale("log")
ax1.set_title("Fig. 6g")

unique_c = df_cst.c.unique()
mfpt_c0 = df_cst[df_cst["c"] == np.inf]["mrdt"] #Case where there is no therapy switching

#Plot for the case where there is no therapy switching/infinite cost parameter
colors_c = cmap(np.linspace(0, 1, len(unique_c)))

fig2, ax2 = plt.subplots(figsize=(8.5,5))

ax2.axhline(mfpt_c0.values[0]/365, lw = 3, color = colors_l[0], label = r"$c=\infty$")

for c in range(len(unique_c)):
    if unique_c[c] < np.inf:
        df_c = df_cst[df_cst["c"] == unique_c[c]].sort_values("tau")
        df_c2= df_c[df_c["plot"] == 1]
        ax2.errorbar(df_c2["tau"], df_c2["mrdt"]/365, fmt = "s", markersize = 7, capsize = 7, yerr=1.96*df_c2["mrdt"]/365, color = colors_c[c+1])
        ax2.plot(df_c["tau"], df_c["mrdt"]/365, label = r"$c=$"+str(unique_c[c]), color = colors_c[c+1], lw = 3)
        
ax2.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax2.tick_params(axis="y", labelsize=30, length = 10, width = 3)
ax2.set_xlabel(r"$\tau$ (yr)", fontsize = 30)
ax2.set_ylabel(r"$\langle T \vert \eta \rangle$ (yr)", fontsize = 30)
ax2.legend(frameon=False, fontsize = 25, handlelength=1, loc = "upper left")
ax2.set_xscale("log")
ax2.set_title("Fig. 6h")