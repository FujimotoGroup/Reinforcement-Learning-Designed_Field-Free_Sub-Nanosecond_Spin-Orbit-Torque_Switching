import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # 非対話型バックエンドを指定
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

from modules import common

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman"
})
#mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams["font.size"] = 24
#plt.rc('text.latex', preamble=r'\usepackage{bm}')

def main():
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)   # 結果を保存するディレクトリ

    load_dir = "../data/100x50x1/aG0.010/M750/J03.0e10_T0/"
    config, t, m, j = common.load(load_dir)

    t_ticks = [0, 0.2, 0.4, 0.6, 0.8]
    j_ticks = [0, 1, 2, 3, 4]

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])
#    common.plot_sphere(ax)
    ax.set_xlabel("$m_x$")
    ax.set_ylabel("$m_y$")
    ax.set_zlabel("$m_z$")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)

    common.plot_sphere_with_depth_wire(ax, elev=10, azim=65)

    ax.plot(m[:, 0], m[:, 1], m[:, 2], color = "red")
#    common.arrow(ax, direction=m[20,:], color='green', alpha=0.8)
    fig.tight_layout()
    ax.view_init(elev=10, azim=65)
#    plt.show()
    plt.savefig(save_dir+"fig1_trajectory.pdf")
    plt.close()

    fig, axes = plt.subplots(2,1,figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(top=0.98, bottom=0.12, left=0.2, right=0.96, hspace=0.05)
    axes = axes.flatten()
    for ax in axes:
        ax.set_xlim([-0.01,t_ticks[-1]])
    axes[0].set_xticks(t_ticks)
    axes[0].tick_params(labelbottom=False)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].set_ylabel("magnetization")
    axes[0].set_yticks([-1,-0.5,0,0.5,1])
    axes[0].plot(t, m[:,0], label="$m_x$", ls='-', color='tab:red')
    axes[0].plot(t, m[:,1], label="$m_y$", ls=':', color='tab:blue')
    axes[0].plot(t, m[:,2], label="$m_z$", ls='-.', color='tab:green')
    axes[0].legend(loc="upper right", labelspacing=0.2)

    axes[1].set_xlabel("$t~(\mathrm{ns})$")
    axes[1].set_xticks(t_ticks)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].set_ylim([-0.1,j_ticks[-1]*1.2])
    axes[1].set_yticks(j_ticks)
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].set_ylabel("$j_e~(\mathrm{MA/cm^2})$", labelpad=17)
    axes[1].plot(t, j, color='black')

#    plt.show()
    plt.savefig(save_dir+"fig1_m-j.pdf")
    plt.close()

if __name__ == '__main__':
    main()
