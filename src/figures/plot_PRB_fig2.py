import toml
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman"
})
#mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams["font.size"] = 20
plt.rc('text.latex', preamble=r'\usepackage{bm}')

def load(directory):
    config = toml.load(directory+"config.toml")
    dt = config["simulation"]["dt"]
    m0 = np.array(config["simulation"]["m0"])

    t = np.loadtxt(directory+'t_judge.txt')
    t = np.concatenate(([-dt], t))

    j = np.loadtxt(directory+"j_judge.txt")
    j = np.concatenate(([0e0], j))

    m = np.loadtxt(directory+"m_judge.txt")
    m = np.concatenate(([m0], m))

    return t, m, j

def main():
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(9, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.11, hspace=0.2, left=0.1, right=0.97, top=0.92, bottom=0.13)

    t_ticks = [0, 0.5, 1.0, 1.5, 2.0]
    m_ticks = [-1, -0.5, 0, 0.5, 1]
    j_ticks = [0, 2, 4, 6, 8]

    labels = ['$m_x$', '$m_y$', '$m_z$']
    colors = ['tab:red', 'tab:blue', 'tab:green']
    lss = ['-', ':', '-.']

    load_dir = "../../data/100x50x1/aG0.010/M2250/J07.0e10_T0/"
    t, m, j = load(load_dir)

    ax_j = fig.add_subplot(gs[1])
    ax_j.set_xlim([-0.01,t_ticks[-1]])
    ax_j.set_xticks(t_ticks)
    ax_j.set_xlabel("$t~(\mathrm{ns})$")
    ax_j.set_xlim([-0.01, t_ticks[-1]])
    ax_j.xaxis.set_minor_locator(AutoMinorLocator())
    ax_j.yaxis.set_minor_locator(AutoMinorLocator())
    ax_j.set_ylim(-0.1, j_ticks[-1]*1.1)
    ax_j.set_yticks(j_ticks)
    ax_j.set_title(r"$j_e~(\mathrm{MA/cm^2})$")
    ax_j.plot(t, j, color='black')

    ax_m = fig.add_subplot(gs[0])
    ax_m.set_xlabel("$t~(\mathrm{ns})$")
    ax_m.set_xlim([-0.01, t_ticks[-1]])
    ax_m.xaxis.set_minor_locator(AutoMinorLocator())
    ax_m.yaxis.set_minor_locator(AutoMinorLocator())
    ax_m.set_xticks(t_ticks)
    ax_m.set_ylim([-1,1])
    ax_m.set_ylabel(r"$\boldsymbol{m}$")
    ax_m.set_yticks(m_ticks)

    n = 1
    selection = np.random.choice(len(m), size=n, replace=False)
    for k, (label, color, ls) in enumerate(zip(labels, colors, lss)):
        ax_m.plot(t, m[:,k], label=label, color=color, ls=ls)

    ax_m.legend()

#    plt.show()
    plt.savefig(save_dir+"PRB_fig2.pdf")
    plt.close()

if __name__ == '__main__':
    main()
