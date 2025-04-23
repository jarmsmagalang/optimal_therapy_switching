import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import src.coupled_continuous as coupled

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
runs = 1000
v=-8*10**(-5) #Drift constant
R1=1 #Reflecting circular boundary
R0=0.4 #Absorbing circular boundary

Dvals = [10**(-5),10**(-6)] #Diffusion parameters
NTvals = np.arange(1,6) #Number of simultaneous therapies
DNTvals = [[D, NT] for D in Dvals for NT in NTvals]

mrdtvals_clim = []
srdtvals_clim = []

mrdtvals_ccst = []
srdtvals_ccst = []

for DNT in DNTvals:
    rdtvals_clim = []
    rdtvals_ccst = []
    for _ in range(runs):
        params = [DNT[0],v,(1/3)*(1/365),DNT[1],R1,R0] #Constant threapy switching rate for both
        rdtvals_clim.append(coupled.cont_sim_RDT(params, 0.8, res_type = "limited", AT = 12)) #A_T set at 12, following Eq. 11
        rdtvals_ccst.append(coupled.cont_sim_RDT(params, 0.8, res_type = "costed"))
        
    mrdtvals_clim.append(np.mean(rdtvals_clim))
    srdtvals_clim.append(np.std(rdtvals_clim))
    
    mrdtvals_ccst.append(np.mean(rdtvals_ccst))
    srdtvals_ccst.append(np.std(rdtvals_ccst))

#Keep results in a dataframe
df_cont = pd.DataFrame(data = {"D": [i[0] for i in DNTvals],
                              "NT": [i[1] for i in DNTvals],
                              "mrdt_lim": mrdtvals_clim,
                              "srdt_lim": srdtvals_clim,
                              "mrdt_cst": mrdtvals_ccst,
                              "srdt_cst": srdtvals_ccst})

#Begin plotting
width = 0.25

fig3, ax3 = plt.subplots(figsize=(8.5,5))

xvals_lim = ["1","2","3","4","6",] #Skips 5 because when $A_T = 12$, $\ell$ is equal for both N_T = 5 and N_T = 6, following Eq. 11

Dhigh_lim = df_cont[df_cont["D"]==Dvals[0]]["mrdt_lim"]/365
Dlow_lim = df_cont[df_cont["D"]==Dvals[1]]["mrdt_lim"]/365

#Take approximate 95% confidence
Dhighs_lim = 1.96*df_cont[df_cont["D"]==Dvals[0]]["srdt_lim"]/365
Dlows_lim =  1.96*df_cont[df_cont["D"]==Dvals[0]]["srdt_lim"]/365

ind_lim = np.arange(len(xvals_lim))

ax3.bar(ind_lim, Dhigh_lim , width, label='$D=10^{-5} \, (\mathrm{days}^{-1})$', color = "mediumslateblue", alpha = 0.75, yerr = Dhighs_lim, error_kw=dict(ecolor='blue', lw=2, capsize=13, capthick=2, alpha = 1))
ax3.bar(ind_lim + width, Dlow_lim, width, label='$D=10^{-6} \, (\mathrm{days}^{-1})$', color = "indianred", alpha = 0.75, yerr = Dlows_lim, error_kw=dict(ecolor='red', lw=2, capsize=13, capthick=2, alpha = 1))
ax3.set_xlabel(r'$N_T$', fontsize = 30)
ax3.set_ylabel(r'$\langle T \vert \eta \rangle$ (yr)', fontsize = 30)
ax3.set_xticks(ind_lim + width / 2, xvals_lim)
ax3.set_yticks(np.linspace(0,20,5))
ax3.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax3.tick_params(axis="y", labelsize=30, length = 10, width = 3)
ax3.legend(frameon=False, loc = 2, fontsize = 25)

fig4, ax4 = plt.subplots(figsize=(8.5,5))

xvals_cst = ["1","2","3","4","5",]

Dhigh_cst = df_cont[df_cont["D"]==Dvals[0]]["mrdt_cst"]/365
Dlow_cst = df_cont[df_cont["D"]==Dvals[1]]["mrdt_cst"]/365

#Take approximate 95% confidence
Dhighs_cst = 1.96*df_cont[df_cont["D"]==Dvals[0]]["srdt_cst"]/365
Dlows_cst =  1.96*df_cont[df_cont["D"]==Dvals[0]]["srdt_cst"]/365

ind_cst = np.arange(len(xvals_cst))

ax4.bar(ind_cst, Dhigh_cst , width, label='$D=10^{-5} \, (\mathrm{days}^{-1})$', color = "mediumslateblue", alpha = 0.75, yerr = Dhighs_cst, error_kw=dict(ecolor='blue', lw=2, capsize=13, capthick=2, alpha = 1))
ax4.bar(ind_cst + width, Dlow_cst, width, label='$D=10^{-6} \, (\mathrm{days}^{-1})$', color = "indianred", alpha = 0.75, yerr = Dlows_cst, error_kw=dict(ecolor='red', lw=2, capsize=13, capthick=2, alpha = 1))
ax4.set_xlabel(r'$N_T$', fontsize = 30)
ax4.set_ylabel(r'$\langle T \vert \eta \rangle$ (yr)', fontsize = 30)
ax4.set_xticks(ind_cst + width / 2, xvals_lim)
ax4.set_yticks(np.linspace(0,20,5))
ax4.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax4.tick_params(axis="y", labelsize=30, length = 10, width = 3)
ax4.legend(frameon=False, loc = 2, fontsize = 25)