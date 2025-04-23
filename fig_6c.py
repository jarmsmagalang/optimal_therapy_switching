import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

import src.heterogeneous as heterogeneous

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.linewidth'] = 3

#Initialize random seed
np.random.seed(1000)

#Initialize parameters
N = 30 #Number of mutation states
mu = 0.05 #Mutation probability
sigma = 0.5 #Mutation state transition probability
init_eta = 0.8 #Initial therapy efficacy
r = (1/3)*(1/365) #Inverse of therapy switching rate

#No therapy switching
params0 = [N, mu, sigma, init_eta, 0]
ttraj0, etatraj0, Htraj0, Ltraj0, Itraj0 = heterogeneous.heterogeneous_RDT(params0, trajectory = True)
tnorm0 = [t/365 for t in ttraj0] #Convert time to years
Hnorm0 = [H/Htraj0[0] for H in Htraj0] #Normalize healthy cell population

#With therapy switching
paramsr = [N, mu, sigma, init_eta, r]
ttrajr, etatrajr, Htrajr, Ltrajr, Itrajr = heterogeneous.heterogeneous_RDT(paramsr, trajectory = True)
tnormr = [t/365 for t in ttrajr] #Convert time to years
Hnormr = [H/Htrajr[0] for H in Htrajr] #Normalize healthy cell population

#Begin plotting
cmap = plt.cm.get_cmap("jet_r")
colors = plt.cm.jet(np.linspace(0,1,N))[::-1]

#Plots for case with no resetting
fig10, (ax10, ax20) = plt.subplots(nrows=2, sharex=True, constrained_layout=True)

#Plot for healthy cell and therapy efficacy trajectories
ax10.plot(tnorm0, Hnorm0, "g", label = "$H/H_0$", lw = 5)
ax10.plot(tnorm0, etatraj0, "k", label = "$\\langle \eta \\rangle_L$")
ax10.legend(frameon=False, handlelength=1,  loc = "lower left")

#Plot for latently infected cell population counts
ax20.set_ylabel("L")
ax20.set_xlabel("Time (yrs)")
for i in np.arange(0,N):
    subLval0 = Ltraj0[:,i]
    ax20.plot(tnorm0, subLval0, label = i/N, color = colors[i])
fig10.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax20, label = r"$\eta$")
ax10.set_title("Fig. 6c, no switching")

#Plots for case with resetting
fig1r, (ax1r, ax2r) = plt.subplots(nrows=2, sharex=True, constrained_layout=True)

#Plot for healthy cell and therapy efficacy trajectories
ax1r.plot(tnormr, Hnormr, "g", label = "$H/H_0$", lw = 5)
ax1r.plot(tnormr, etatrajr, "k", label = "$\\langle \eta \\rangle_L$")
ax1r.legend(frameon=False,  handlelength=1,  loc = "lower left")

#Plot for latently infected cell population counts
ax2r.set_ylabel("L")
ax2r.set_xlabel("Time (yrs)")
for i in np.arange(0,N):
    subLvalr = Ltrajr[:,i]
    ax2r.plot(tnormr, subLvalr, label = i/N, color = colors[i])
fig1r.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2r, label = r"$\eta$")
ax1r.set_title("Fig. 6c, switching")
