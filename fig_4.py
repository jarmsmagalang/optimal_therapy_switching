import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import tqdm as tqdm

import src.coupled_continuous as coupled
import src.uncoupled_discrete as uncoupled

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

### Begin for Fig. 4a
D=10**(-4) #Diffusion constant
v=-8*10**(-5) #Drift constant
tau = 3 #Therapy switching rate
r = (1/tau)*(1/365) #Inverse of switching rate

#Switching rate used in Fig 4d
r1 = (1/1)*(1/365) #Fast switching
r10 = (1/10)*(1/365) #Slow switching

NT = 2 # Number of simultaneous therapies
R1=1 #Reflecting circular boundary
R0=0.4 #Absorbing circular boundary

params=[[D,v,r,NT,R1,R0],   #Switching, For Fig. 4a
        [D,v,0,NT,R1,R0],   #No switching, For Fig. 4a and 4d
        [D,v,r1,NT,R1,R0],  #Fast switching, For Fig. 4d
        [D,v,r10,NT,R1,R0]] #Slow switching, For Fig. 4d

#Initialize eta1 and eta2 values
xvals = np.linspace(0,1,1001)
yvals = np.linspace(0,1,1001)
xs = []
ys = []
rs = []

for xv in xvals:
    for yv in yvals:
        rv = np.sqrt(xv**2 + yv**2)
        #Take radii that fall within the reflecting and absorbing boudnaries
        if R0 <= rv < R1:
            xs.append(xv)
            ys.append(yv)
            rs.append(rv)
xs = np.array(xs)
ys = np.array(ys)
rs = np.array(rs)

dt = []

#For Fig 4d
cont_percent_pos = np.zeros(2)

for i in range(len(rs)):
    res = np.real(coupled.cont_ana_MRDT(params[0],rs[i])) #Mean RDT with resetting
    no_res = coupled.cont_ana_MRDT(params[1],rs[i]) #Mean RDT without resetting
    
    res1 = np.real(coupled.cont_ana_MRDT(params[2],rs[i])) #For Fig 4d
    res10 = np.real(coupled.cont_ana_MRDT(params[3],rs[i])) #For Fig 4d
    
    #Compute for the percent of area positive of the coupled continuous model, For Fig 4d
    if res1 > no_res: #Fast switching
        #If resetting is better, increment by inverse of total area of the space
        cont_percent_pos[0] += 1/len(rs)
    if res10 > no_res: #Slow switching
        #If resetting is better, increment by inverse of total area of the space
        cont_percent_pos[1] += 1/len(rs)
    
    #Take difference of case with and without resetting
    dt.append((res-no_res)/365)

#Keep values in dataframe and pivot table
df_cont=pd.DataFrame(data = {"eta1": xs,
                             "eta2": ys,
                             "mrdt_diff": dt})
df_pivot1 = df_cont.pivot_table(index = "eta1", columns = "eta2", values = "mrdt_diff")

### Begin for Figs. 4b and 4c
N = 30 #Number of states in the discrete uncoupled model
rho = int(N*0.4)-1 #Radius of absorption in discrete uncoupled model

#List of all valid states in the discrete uncoupled model
statelist2 = [uncoupled.ind_to_state(i, N) for i in np.arange(0,N**2)]
delstate = [s for s in statelist2 if np.floor(np.sqrt(s[0]**2 + s[1]**2)) <= rho]
statelist = [s for s in statelist2 if s not in delstate]

disc_params=[[N, rho, r, r, v,v,D,D], #Symmetric switching, Used for Fig. 4b
             [N, rho, r, 0, v,v,D,D], #Asymmetric switching, Used for Fig. 4c
             [N, rho, 0, 0, v,v,D,D], #No switching, Used for Fig. 4b, 4c, 4d
             [N, rho, r1, r1, v,v,D,D], #Symmetric fast switching, Used for Fig. 4d
             [N, rho, r1, 0, v,v,D,D], #Asymmetric fast switching, Used for Fig. 4d
             [N, rho, r10, r10, v,v,D,D], #Symmetric slow switching, Used for Fig. 4d
             [N, rho, r10, 0, v,v,D,D], #Symmetric slow switching, Used for Fig. 4d
             ]

#Begin computing for ana. mean RDT for all possible initial positions, for all selected parameters
xs = np.array([s[0] for s in statelist])
ys = np.array([s[1] for s in statelist])

sym_mrdt_diff = [] #For Fig. 4b
asym_mrdt_diff = [] #For Fig. 4c

#For Fig. 4d
sym_percent_pos = np.zeros(2)
asym_percent_pos = np.zeros(2)

for i in tqdm.tqdm(range(len(statelist))):
    sym_res = np.real(uncoupled.disc_ana_MRDT(disc_params[0],statelist[i][0], statelist[i][1]))
    asym_res = np.real(uncoupled.disc_ana_MRDT(disc_params[1],statelist[i][0], statelist[i][1]))
    no_res = uncoupled.disc_ana_MRDT(disc_params[2], statelist[i][0], statelist[i][1])
    
    sym_mrdt_diff.append((sym_res-no_res)/365) #Symmetric difference
    asym_mrdt_diff.append((asym_res-no_res)/365) #Asymmetric difference
    
    sym_res1 = np.real(uncoupled.disc_ana_MRDT(disc_params[3],statelist[i][0], statelist[i][1]))
    asym_res1 = np.real(uncoupled.disc_ana_MRDT(disc_params[4],statelist[i][0], statelist[i][1]))
    
    sym_res10 = np.real(uncoupled.disc_ana_MRDT(disc_params[5],statelist[i][0], statelist[i][1]))
    asym_res10 = np.real(uncoupled.disc_ana_MRDT(disc_params[6],statelist[i][0], statelist[i][1]))
    
    if sym_res1 > no_res:
        #For symmetric fast switching
        #If resetting is better, increment by inverse of total area of the space
        sym_percent_pos[0] += 1/len(statelist)
    if sym_res10 > no_res:
        #For symmetric slow switching
        #If resetting is better, increment by inverse of total area of the space
        sym_percent_pos[1] += 1/len(statelist)
        
    if asym_res1 > no_res:
        #For asymmetric fast switching
        #If resetting is better, increment by inverse of total area of the space
        asym_percent_pos[0] += 1/len(statelist)
    if asym_res10 > no_res:
        #For asymmetric slow switching
        #If resetting is better, increment by inverse of total area of the space
        asym_percent_pos[1] += 1/len(statelist)
        
#Keep values in dataframe and pivot table
df_disc=pd.DataFrame(data = {"eta1": xs,
                        "eta2": ys,
                        "sym_mrdt_diff": sym_mrdt_diff,
                        "asym_mrdt_diff": asym_mrdt_diff})
df_pivot2 = df_disc.pivot_table(index = "eta1", columns = "eta2", values = "sym_mrdt_diff") #Symmetric switching
df_pivot3 = df_disc.pivot_table(index = "eta1", columns = "eta2", values = "asym_mrdt_diff") #Asymmetric switching


### Begin plotting
cmaps = sns.color_palette("bwr", as_cmap=True)

#Contour plot for coupled continuous model
fig1, ax1 = plt.subplots()
divnorm1 = TwoSlopeNorm(vcenter=0, vmax=df_cont["mrdt_diff"].max(), vmin=-df_cont["mrdt_diff"].max(), )

eta_x1, eta_y1 = np.meshgrid(df_pivot1.index.to_numpy(), df_pivot1.columns.to_numpy())

ax1.fill_between([0,1],[0,0],[1,1], facecolor="none", hatch="xx", edgecolor="black", linewidth=0.0, alpha = 0.2)
cs = ax1.contourf(eta_x1, eta_y1, df_pivot1.values, 9, cmap = cmaps, norm=divnorm1) 
ax1.contour(eta_x1, eta_y1, df_pivot1.values,[0],linestyles='dashed', linewidths = 5, colors = "black")
ax1.set_xlabel(r"$\eta_{1}$", fontsize = 30)
ax1.set_ylabel(r"$\eta_{2}$", fontsize = 30)
ax1.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax1.tick_params(axis="y", labelsize=30, length = 10, width = 3)
cbar = fig1.colorbar(cs)
cbar.set_label(r'$\Delta T$ (yr)', fontsize = 30)
cbar.ax.tick_params(labelsize=30, length = 10, width = 3) 

ax1.set_xticks(np.linspace(0,1,3))
ax1.set_yticks(np.linspace(0,1,3))

ax1.set_xlim((0,1.0))
ax1.set_ylim((0,1.0))
ax1.set_aspect('equal')
ax1.set_title("Fig. 4a")
fig1.tight_layout()

#Contour plot for uncoupled discrete model with symmetric switching
fig2, ax2 = plt.subplots()

divnorm2 = TwoSlopeNorm(vcenter=0, vmax=df_disc["sym_mrdt_diff"].max(), vmin=-df_disc["sym_mrdt_diff"].max(), )

eta_x2, eta_y2 = np.meshgrid(df_pivot2.index.to_numpy()/30, df_pivot2.columns.to_numpy()/30)

ax2.fill_between([0,1],[0,0],[1,1], facecolor="none", hatch="xx", edgecolor="black", linewidth=0.0, alpha = 0.2)

cs = ax2.contourf(eta_x2, eta_y2, df_pivot2.values, 13, cmap = cmaps,norm=divnorm2) 
ax2.contour(eta_x2, eta_y2, df_pivot2.values,[0],linestyles='dashed', linewidths = 5, colors = "black")
ax2.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax2.tick_params(axis="y", labelsize=30, length = 10, width = 3)

cbar2 = fig2.colorbar(cs)
cbar2.set_label(r'$\Delta T$ (yr)', fontsize = 30)
cbar2.ax.tick_params(labelsize=30, length = 10, width = 3) 

ax2.set_xticks(np.linspace(0,1,3))
ax2.set_yticks(np.linspace(0,1,3))
ax2.set_xlabel(r"$\eta_{1}$", fontsize = 30)
ax2.set_ylabel(r"$\eta_{2}$", fontsize = 30)
ax2.set_xlim((0,1.0))
ax2.set_ylim((0,1.0))
ax2.set_aspect('equal')
ax2.set_title("Fig. 4b")
fig2.tight_layout()

#Contour plot for uncoupled discrete model with asymmetric switching
fig3, ax3 = plt.subplots()

divnorm3 = TwoSlopeNorm(vcenter=0, vmax=df_disc["asym_mrdt_diff"].max(), vmin=-df_disc["asym_mrdt_diff"].max(), )

eta_x3, eta_y3 = np.meshgrid(df_pivot3.index.to_numpy()/30, df_pivot3.columns.to_numpy()/30)

ax3.fill_between([0,1],[0,0],[1,1], facecolor="none", hatch="xx", edgecolor="black", linewidth=0.0, alpha = 0.2)

cs = ax3.contourf(eta_x3, eta_y3, df_pivot3.values, 13, cmap = cmaps,norm=divnorm3) 
ax3.contour(eta_x3, eta_y3, df_pivot3.values,[0],linestyles='dashed', linewidths = 5, colors = "black")
ax3.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax3.tick_params(axis="y", labelsize=30, length = 10, width = 3)

cbar3 = fig3.colorbar(cs)
cbar3.set_label(r'$\Delta T$ (yr)', fontsize = 30)
cbar3.ax.tick_params(labelsize=30, length = 10, width = 3) 

ax3.set_xticks(np.linspace(0,1,3))
ax3.set_yticks(np.linspace(0,1,3))
ax3.set_xlabel(r"$\eta_{1}$", fontsize = 30)
ax3.set_ylabel(r"$\eta_{2}$", fontsize = 30)
ax3.set_xlim((0,1.0))
ax3.set_ylim((0,1.0))
ax3.set_aspect('equal')
ax3.set_title("Fig. 4c")
fig3.tight_layout()

#Bar plot for counting the probability that switching is more beneficial for all models considered
bartickvals = [1,10]

df_barplot = pd.DataFrame(data={"(a)": cont_percent_pos,
                                "(b)": sym_percent_pos,
                                "(c)": asym_percent_pos,
                                }, index = bartickvals)

fig4, ax4 = plt.subplots()
df_barplot.plot(ax = ax4, kind = "bar", rot=0, color= ['#377eb8', '#e41a1c', '#4daf4a'], alpha =0.75)
ax4.set_ylim(0,1)
ax4.set_xlabel(r"$\tau$ (yr)", fontsize = 30)
ax4.set_ylabel(r"$P$", fontsize = 30)
ax4.legend(frameon=False, loc = 1, fontsize = 20)
ax4.tick_params(axis="x", labelsize=30, length = 10, width = 3)
ax4.tick_params(axis="y", labelsize=30, length = 10, width = 3)
# ax4.set_title("Fig. 4d")
fig4.tight_layout()
