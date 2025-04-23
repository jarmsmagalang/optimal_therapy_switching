import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patches as patches

import src.coupled_continuous as coupled
import src.uncoupled_discrete as uncoupled
import src.heterogeneous as heterogeneous
import src.ana_moments as analytic
import tqdm as tqdm

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams['axes.linewidth'] = 3

#Initialize parameters
N = 30 #Number of mutation states
mu = 0.05 #Mutation probability
sigma = 0.5 #Mutation state transition probability
init_eta = 0.8 #Initial therapy efficacy

runs = 10000

params0 = [N, mu, sigma, init_eta, 0]

#Simulate heterogeneous model RDTs
het_rdt = [heterogeneous.heterogeneous_RDT(params0, trajectory = False) for _ in tqdm.tqdm(range(runs))]

#Compute for mean RDT of the heterogeneous model
het_mrdt = np.nanmean(het_rdt)

#Ideal search parameters have been inferred by searching various values of v and D
v_search = np.linspace(-2.5e-4,0,1001)
D_search = np.linspace(0,1.5e-4,1001)

search_input = []

for vs in v_search:
    for Ds in D_search:
        #Compute for absolute mean difference between heterogeneous mean RDT and the analytical mean RDT of the drift-diffusion process
        lossfcn, eul_var = analytic.loss_fcn([vs, Ds], het_mrdt)
        search_input.append([vs, Ds, lossfcn, eul_var])
        
df_search = pd.DataFrame(data=search_input, columns = ["v", "D", "loss", "var"])

#Obtain only the values that has both an absolute mean difference close to zero and has the smallest analytical variance
df_threshloss = df_search[df_search["loss"] <= 1].nsmallest(1, "var")

#Optimal parameters for drift and diffusion
v_minvar = df_threshloss.nsmallest(1, "var")["v"].values[0]
D_minvar = df_threshloss.nsmallest(1, "var")["D"].values[0]

#Generate RDTs for the continuous and discrete models, used for Fig. 6e
cont_rdt = [coupled.cont_sim1D_RDT([D_minvar, v_minvar, 0, 1, 0], init_eta) for _ in range(runs)] #Continuous
disc_rdt = [uncoupled.disc_sim1D_RDT([v_minvar, D_minvar, 30, 0], init_eta) for _ in range(runs)] #Discrete

#Generate trajectories for all models, used for Fig. 
het_ttraj, het_etatraj, _, _, _ = heterogeneous.heterogeneous_RDT(params0, trajectory = True)
cont_ttraj, cont_etatraj = coupled.cont_sim1D_RDT([D_minvar, v_minvar, 0, 1, 0], init_eta, trajectory = True)
disc_ttraj, disc_etatraj = uncoupled.disc_sim1D_RDT([v_minvar, D_minvar, 30, 0], init_eta, trajectory = True)

#Convert search dataframe into pivot table
df_search_loss_pivot = df_search.pivot_table(values = "loss", columns = "v", index = "D")

#Relabel pivot table columns and rows to remove round-off errors
df_search_loss_pivot.columns = pd.Index([np.round(i*(10**8))/(10**8) for i in df_search_loss_pivot.columns], name="v")
df_search_loss_pivot.index = pd.Index([np.round(i*(10**8))/(10**8) for i in df_search_loss_pivot.index], name="D")
df_search_loss_pivot = df_search_loss_pivot.sort_index(ascending = False)

#Begin plotting

# Customize the x-axis tick labels
v_search_labels = [str(round(x*(10**4),2)) for x in np.linspace(-2.5e-4,0,3)]

# # Customize the y-axis tick labels
D_search_labels = [str(round(x*(10**4),2)) for x in np.linspace(0,1.5e-4,3)[::-1]]

#Plots for Fig. 6d
fig1, ax1 = plt.subplots()
ax1 = sns.heatmap(df_search_loss_pivot, square = True, clip_on = False, norm=LogNorm())

#Initialize colorbar
cbar1 = ax1.collections[0].colorbar
cbar1.set_label(r"$\Delta T$", fontsize = 15)
cbar1.ax.tick_params(labelsize=15, length = 10, width = 3) 
#Include border around colorbar
cbar1.ax.spines["outer"] = patches.Rectangle(
    (0, 0), 1, 1, transform=cbar1.ax.transAxes,  # Full coverage of the colorbar
    linewidth=3, edgecolor="black", facecolor="none"
)

# Get heatmap dimensions
xlim = ax1.get_xlim()  # Get x-axis limits
ylim = ax1.get_ylim()  # Get y-axis limits

# Include border around heatmap
expand = 0.5
rect = patches.Rectangle(
    (xlim[0] - expand, ylim[0] - expand),  # Bottom-left corner (slightly outside)
    (xlim[1] - xlim[0]) + 2 * expand,  # Expanded width
    (ylim[1] - ylim[0]) + 2 * expand,  # Expanded height
    linewidth=3, edgecolor='black', facecolor='none', clip_on=False
)
ax1.add_patch(rect)

#Include offset text for scientific notation of the axes
offset_text = r"$\times 10^{-4}$"
# Position the offset near the top of the x-axis
ax1.text(0.15, 1., offset_text, transform=ax1.transAxes, fontsize=15, ha='right', va='bottom')
# Position the offset near the right of the y-axis
ax1.text(0.9, -0.15, offset_text, transform=ax1.transAxes, fontsize=15, ha='left', va='top')

#Replace ticklabels
ax1.set_xticks(np.linspace(0, df_search_loss_pivot.shape[0], len(v_search_labels)), v_search_labels, rotation=0)
ax1.set_yticks(np.linspace(0, df_search_loss_pivot.shape[1], len(D_search_labels)), D_search_labels)
ax1.xaxis.set_tick_params(length = 10, width = 3)
ax1.yaxis.set_tick_params(length = 10, width = 3)

#Plot the point corresponding to the optimal v and D
ax1.plot(df_search_loss_pivot.columns.get_loc(v_minvar),df_search_loss_pivot.index.get_loc(D_minvar), "o", markersize = 10, c = "white", markeredgewidth = 2, markeredgecolor = "black")
ax1.set_title("Fig. 1d")

#Plots for Fig. 6e
fig2, ax2 = plt.subplots()
ax2.hist([i/365 for i in het_rdt], alpha = 0.4, density = True, bins = "auto", label = "Heterogenous", color = "blue")
ax2.hist([i/365 for i in cont_rdt], alpha = 0.5, density = True, bins = "auto", label = "Coupled", color = "orange")
ax2.hist([i/365 for i in disc_rdt], alpha = 0.4, density = True, bins = "auto", label = "Uncoupled", color = "green")

ax2.legend(frameon=False, handlelength=1)
ax2.tick_params(axis="x", labelsize=15, length = 10, width = 3)
ax2.tick_params(axis="y", labelsize=15, length = 10, width = 3)
ax2.set_xlabel("RDT (yr)")
ax2.set_ylabel("Density")
ax2.set_title("Fig. 6e")

#Plots for Fig. 6f
fig3, ax3 = plt.subplots()
ax3.plot([i/365 for i in het_ttraj], het_etatraj, color = "blue", label = "Heterogeneous")
ax3.plot([i/365 for i in cont_ttraj], cont_etatraj, color = "orange", label = "Coupled")
ax3.plot([i/365 for i in disc_ttraj], disc_etatraj, color = "green", label = "Uncoupled")
ax3.set_xlabel("Time (yr)")
ax3.set_ylabel(r"$\eta$")
ax3leg = ax3.legend(frameon=False, markerscale=10, handlelength=1)
for legobj in ax3leg.legend_handles :
    legobj.set_linewidth(5)
ax3.tick_params(axis="x", labelsize=15, length = 10, width = 3)
ax3.tick_params(axis="y", labelsize=15, length = 10, width = 3)
ax3.set_ylim(0,1)