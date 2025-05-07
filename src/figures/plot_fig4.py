import glob
import re
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib.animation as animation

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

def load(directory): # {{{
    def extract_mj_key(path):
        m_match = re.search(r"M(\d+)", path)
        j_match = re.search(r"J([0-9.eE+-]+)", path)
        m_val = int(m_match.group(1)) if m_match else 0
        j_val = float(j_match.group(1)) if j_match else 0.0
        return (m_val, j_val)

    paths = glob.glob(directory+"M*/J*/", recursive=False)
    sorted_paths = sorted(paths, key=extract_mj_key)
    return sorted_paths
# }}}

def plot_each_case(directory): # {{{
    t = np.loadtxt(directory+'t.txt') * 1e9
    j = np.loadtxt(directory+'j.txt') * 1e-10
    m = np.loadtxt(directory+'m.txt')

    Ms = re.search(r"M=(\d+)", directory).group(1)

    history = np.loadtxt(directory+"reward history.txt")
    label_b = 'for each episode'
    episodes = np.arange(len(history))

    slice_num = 20
    average = [ history[i:i+slice_num].mean() for i in episodes[::slice_num]]
    label_m = f'Average rewards for {slice_num} episodes'

    window = 10
    weights = np.ones(window) / window
    moving_avg = np.convolve(history, weights, mode='valid')
    label_ma = f'Moving average\nover {window} episodes'

    t_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    m_ticks = [-1, -0.5, 0, 0.5, 1]

    fig, axes = plt.subplots(1,3,figsize=(18, 7))
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.03, right=0.97, wspace=0.2)
    axes = axes.flatten()

    axes[0].set_title("Rewards (a.u.)")
    axes[0].set_xlim([0, len(history)])
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].set_xlabel("episode")
    axes[0].tick_params(labelleft=False)
    axes[0].axvspan(975, 1000, color='yellow', alpha=0.5)
    axes[0].plot(episodes, history, color="tab:gray")
    axes[0].plot(moving_avg, color="black")

    axes[1].set_xlim([-0.01, t_ticks[-1]])
    axes[1].set_xticks(t_ticks)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].set_ylim([-1,1])
    axes[1].set_yticks(m_ticks)
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].plot(t, m[:,0], label="$m_x$", ls='-', color='tab:red')
    axes[1].plot(t, m[:,1], label="$m_y$", ls=':', color='tab:blue')
    axes[1].plot(t, m[:,2], label="$m_z$", ls='-.', color='tab:green')
    axes[1].legend(loc="upper right")
    text_Ms = axes[1].text(0.45, 1.015, r"$M_{\mathrm{s}} = "+Ms+"~\mathrm{kA/m}$", transform=axes[1].transAxes)

    axes[2].set_title("$j_e~(\mathrm{MA/cm^2})$")
    axes[2].set_xlim([-0.01,0.5])
    axes[2].set_xticks(t_ticks)
    axes[2].set_ylim([-0.1,12])
    axes[2].xaxis.set_minor_locator(AutoMinorLocator())
    axes[2].yaxis.set_minor_locator(AutoMinorLocator())
    axes[2].tick_params(labelbottom=False)
    axes[2].plot(t, j, color='black')

    axes[1].tick_params(labelbottom=True)
    axes[1].set_xlabel("$t~(\mathrm{ns})$")
    axes[1].set_xticks(t_ticks)
    axes[2].tick_params(labelbottom=True)
    axes[2].set_xlabel("$t~(\mathrm{ns})$")
    axes[2].set_xticks(t_ticks)

    plt.show()
    plt.close()
# }}}

def plot_phase_map(load_dir): # {{{
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)

    data = np.loadtxt(load_dir+"scatter_data.txt")

    M = data[:, 0]*1e-3
    j = data[:, 1]*1e-10
    r = data[:, 2]

    select = (M == 750) & (j == 3e0)
    success = r > 0
    failure = r == 0

    j_ticks = [0, 5, 10]

    fig, ax = plt.subplots(1,1,figsize=(10, 7))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.1, right=0.98)
    ax.set_ylim([0.1, 11.5])
    ax.set_yticks(j_ticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel("$M_{\mathrm{s}}~(\mathrm{kA/m}$)")
    ax.set_ylabel("$j_e$ (MA/cm$^2$)")
    ax.scatter(
        M[success], j[success],
        c='tab:blue', marker='o', label='Success',
        edgecolors='k', linewidths=0.5, s=90
    )
    ax.scatter(
        M[failure], j[failure],
        c='tab:red', marker='x', label='Failure',
        linewidths=1.2, s=90
    )
    ax.scatter(
        M[select], j[select],
        c='yellow', marker='*',
        edgecolors='k', linewidths=0.5, s=120
    )
    legend = ax.legend(title="", loc='lower right', frameon=True, handletextpad=0, borderaxespad=0.4, borderpad=0.4)
    legend.get_frame().set_alpha(0.95)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.text(0.9, 1.01, r"$T = 0$", transform=ax.transAxes)

#    plt.show()
    plt.savefig(save_dir+"fig4.pdf", format='pdf')
#    plt.savefig("fig4.png")
    plt.close()
# }}}

def make_anime(path): # {{{
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)

    # init {{{
    directory = path[0]
    config, t, m, j = common.load(directory)
    Ms = f'{config["material"]["M"]*1e-3:.0f}'

    history = np.loadtxt(directory+"reward_history.txt")
    label_b = 'for each episode'
    episodes = np.arange(len(history))

    slice_num = 20
    average = [ history[i:i+slice_num].mean() for i in episodes[::slice_num]]
    label_m = f'Average rewards for {slice_num} episodes'

    window = 10
    weights = np.ones(window) / window
    moving_avg = np.convolve(history, weights, mode='valid')
    episodes_valid = episodes[window-1:]
    label_ma = f'Moving average\nover {window} episodes'

    t_ticks = [0, 0.2, 0.4, 0.6, 0.8]
    m_ticks = [-1, -0.5, 0, 0.5, 1]

    fig, axes = plt.subplots(1,3,figsize=(18, 7))
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.03, right=0.97, wspace=0.2)
    axes = axes.flatten()

    axes[0].set_title("Rewards (a.u.)")
    axes[0].set_xlim([0, len(history)])
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].set_xlabel("episode")
    axes[0].tick_params(labelleft=False)
    axes[0].axvspan(975, 1000, color='yellow', alpha=0.5)
    plot_history_raw, = axes[0].plot(episodes, history, color="tab:gray")
    plot_history_ma , = axes[0].plot(episodes_valid, moving_avg, color="black")

    axes[1].set_xlim([-0.01, t_ticks[-1]])
    axes[1].set_xticks(t_ticks)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].set_ylim([-1,1])
    axes[1].set_yticks(m_ticks)
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    plot_mx, = axes[1].plot(t, m[:,0], label="$m_x$", ls='-', color='tab:red')
    plot_my, = axes[1].plot(t, m[:,1], label="$m_y$", ls=':', color='tab:blue')
    plot_mz, = axes[1].plot(t, m[:,2], label="$m_z$", ls='-.', color='tab:green')
    axes[1].legend(loc="upper right")
    text_Ms = axes[1].text(0.45, 1.015, r"$M_{\mathrm{s}} = "+Ms+"~\mathrm{kA/m}$", transform=axes[1].transAxes)

    axes[2].set_title("$j_e~(\mathrm{MA/cm^2})$")
    axes[2].set_xlim([-0.01,0.5])
    axes[2].set_xticks(t_ticks)
    axes[2].set_ylim([-0.1,12])
    axes[2].xaxis.set_minor_locator(AutoMinorLocator())
    axes[2].yaxis.set_minor_locator(AutoMinorLocator())
    axes[2].tick_params(labelbottom=False)
    plot_j, = axes[2].plot(t, j, color='black')

    axes[1].tick_params(labelbottom=True)
    axes[1].set_xlabel("$t~(\mathrm{ns})$")
    axes[1].set_xticks(t_ticks)
    axes[2].tick_params(labelbottom=True)
    axes[2].set_xlabel("$t~(\mathrm{ns})$")
    axes[2].set_xticks(t_ticks)
    # }}}

    def update_anime(num): # {{{
#        if (num%10 == 0): print(num, end=', ', flush=True)

        directory = path[num]
        config, t, m, j = common.load(directory)

        Ms = f'{config["material"]["M"]*1e-3:.0f}'

        history = np.loadtxt(directory+"reward_history.txt")
        moving_avg = np.convolve(history, weights, mode='valid')
        episodes = np.arange(len(history))
        episodes_valid = episodes[window-1:]

        text_Ms.set_text(r"$M_{\mathrm{s}} = "+Ms+"~\mathrm{kA/m}$")

        plot_history_raw.set_data(episodes, history)
        plot_history_ma.set_data(episodes_valid, moving_avg)
        axes[0].relim()
        axes[0].autoscale_view()

        plot_mx.set_data(t, m[:,0])
        plot_my.set_data(t, m[:,1])
        plot_mz.set_data(t, m[:,2])

        plot_j.set_data(t, j)

        return []
    # }}}

    anime = animation.FuncAnimation(fig, update_anime, frames=np.arange(0, len(path)), interval=200, repeat=False, cache_frame_data=True, blit=True)
    anime.save(save_dir+"fig4_anime.mp4", writer="ffmpeg", dpi=200)
# }}}

def main():
    load_dir = "../data/100x50x1/aG0.010/"
    plot_phase_map(load_dir)

    path = load(load_dir)
    make_anime(path)

if __name__ == '__main__':
    main()
