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

def main():
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)

    pickup_episodes = [20, 200, 730]
    n = len(pickup_episodes) + 1

    load_dir = "../data/100x50x1/aG0.010/M750/J03.0e10_T0/"

    config = toml.load(load_dir+"config.toml")
    dt = config["simulation"]["dt"]
    m0 = np.array(config["simulation"]["m0"])

    t = np.loadtxt(load_dir+'t.txt')
    t = np.concatenate(([-dt], t))

    j = []
    m = []
    for i in pickup_episodes:
        tmp_j = np.loadtxt(load_dir+f'j_episode{i:05d}.txt')
        tmp_j = np.concatenate(([0e0], tmp_j))
        tmp_m = np.loadtxt(load_dir+f'm_episode{i:05d}.txt')
        tmp_m = np.concatenate(([m0], tmp_m))
        j.append(tmp_j)
        m.append(tmp_m)
    tmp_j = np.loadtxt(load_dir+f'j_best.txt')
    tmp_j = np.concatenate(([0e0], tmp_j))
    tmp_m = np.loadtxt(load_dir+f'm_best.txt')
    tmp_m = np.concatenate(([m0], tmp_m))
    j.append(tmp_j)
    m.append(tmp_m)

    history = np.loadtxt(load_dir+'reward_history.txt')
    label_b = 'for each episode'
    episodes = np.arange(len(history))

#    slice_num = 20
#    average = [ history[i:i+slice_num].mean() for i in episodes[::slice_num]]
#    label_m = f'Average rewards for {slice_num} episodes'

    window = 10
    weights = np.ones(window) / window
    moving_avg = np.convolve(history, weights, mode='valid')
    label_ma = f'Moving average\nover {window} episodes'

    t_ticks = [0, 0.2, 0.4, 0.6, 0.8]
    j_ticks = [0, 1, 2, 3]

    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(n, 3, figure=fig, width_ratios=[1.8, 1, 1], wspace=0.2, hspace=0.2, left=0.02, right=0.98, top=0.93, bottom=0.12)

    ax_history = fig.add_subplot(gs[0:n, 0])
    ax_history.set_title("Rewards (a.u.)")
    ax_history.set_xlim([0, len(history)])
    ax_history.set_xlabel("episode")
    ax_history.tick_params(labelleft=False)
    ax_history.axvspan(975, 1000, color='yellow', alpha=0.5)
    ax_history.plot(episodes, history, color="tab:gray")
    ax_history.plot(moving_avg, color="black")
#    ax_history.plot(episodes[::slice_num], average, color="black", ls=':')
#    ax_history.legend()
    k = 220+np.argsort(history[220:500])[1]
    ax_history.annotate(
        label_b,
        xy=(k, history[k]),
        xytext=(k+120, history[k]),
        arrowprops=dict(arrowstyle='->', color='tab:gray', lw=1),
        va='center',
        color="tab:gray"
    )
    k = 100+np.argsort(moving_avg[100:800])[12]
    ax_history.annotate(
        label_ma,
        xy=(k, moving_avg[k]),
        xytext=(k+320, moving_avg[k]),
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
        va='center',
        color="black"
    )
    for i, epi in enumerate(pickup_episodes):
        episode = epi - 1
        ax_history.annotate(
            f"{epi}th",
            xy=(episode, history[episode]),
            xytext=(episode+110, history[episode]),
            arrowprops=dict(arrowstyle='simple', color='tab:red', lw=1),
            va='center',
            color="tab:red"
        )

    for i, episode in enumerate(pickup_episodes):
        ax_m = fig.add_subplot(gs[i, 1])
        ax_m.set_xlim([-0.01, t_ticks[-1]])
        ax_m.xaxis.set_minor_locator(AutoMinorLocator())
        ax_m.yaxis.set_minor_locator(AutoMinorLocator())
        ax_m.set_xticks(t_ticks)
        ax_m.tick_params(labelbottom=False)
        ax_m.set_ylim([-1,1])
        ax_m.plot(t, m[i][:,0], label="$m_x$", ls='-', color='tab:red')
        ax_m.plot(t, m[i][:,1], label="$m_y$", ls=':', color='tab:blue')
        ax_m.plot(t, m[i][:,2], label="$m_z$", ls='-.', color='tab:green')
        position_x = 0.63 if i >= 2 else 0.01
        ax_m.text(position_x, 0.05, f"{episode}th", transform=ax_m.transAxes)
        ax_j = fig.add_subplot(gs[i, 2])
        ax_j.set_xlim([-0.01,0.5])
        ax_j.set_xticks(t_ticks)
        ax_j.xaxis.set_minor_locator(AutoMinorLocator())
        ax_j.tick_params(labelbottom=False)
        ax_j.set_ylim([-0.1,j_ticks[-1]*1.2])
        ax_j.set_yticks(j_ticks)
        ax_j.plot(t, j[i], color='black')
        if i == 0:
            ax_m.set_title(r"$\boldsymbol{m}$")
            ax_j.set_title("$j_e~(\mathrm{MA/cm^2})$")
            t_x = 200
            ax_m.annotate(
                "$m_x$",
                xy=(t[t_x], m[i][t_x,0]),
                xytext=(t[t_x+80], m[i][t_x,0]-0.1),
                arrowprops=dict(
                    arrowstyle='->',
                    color='tab:red',
                    lw=1,
                    mutation_scale=10
                ),
                va='center',
                color="tab:red"
            )
            t_y = 700
            ax_m.annotate(
                "$m_y$",
                xy=(t[t_y], m[i][t_y,1]),
                xytext=(t[t_y], m[i][t_y,1]-1.1),
                arrowprops=dict(
                    arrowstyle='->',
                    color='tab:blue',
                    lw=1,
                    mutation_scale=10
                ),
                ha='center',
                color="tab:blue"
            )
            t_z = 350
            ax_m.annotate(
                "$m_z$",
                xy=(t[t_z], m[i][t_z,2]),
                xytext=(t[t_z], m[i][t_z,2]-0.8),
                arrowprops=dict(
                    arrowstyle='->',
                    color='tab:green',
                    lw=1,
                    mutation_scale=10
                ),
                ha='center',
                color="tab:green"
            )

    ax_m = fig.add_subplot(gs[-1, 1])
    ax_m.set_xlim([-0.01, t_ticks[-1]])
    ax_m.xaxis.set_minor_locator(AutoMinorLocator())
    ax_m.yaxis.set_minor_locator(AutoMinorLocator())
    ax_m.set_xticks(t_ticks)
    ax_m.tick_params(labelbottom=False)
    ax_m.set_ylim([-1,1])
    ax_m.plot(t, m[-1][:,0], label="$m_x$", ls='-', color='tab:red')
    ax_m.plot(t, m[-1][:,1], label="$m_y$", ls=':', color='tab:blue')
    ax_m.plot(t, m[-1][:,2], label="$m_z$", ls='-.', color='tab:green')
    ax_m.text(0.57, 0.06, f"highest", transform=ax_m.transAxes)
    ax_j = fig.add_subplot(gs[-1, 2])
    ax_j.set_xlim([-0.01,0.5])
    ax_j.set_xticks(t_ticks)
    ax_j.set_ylim([-0.1,j_ticks[-1]*1.2])
    ax_j.set_yticks(j_ticks)
    ax_j.xaxis.set_minor_locator(AutoMinorLocator())
    ax_j.tick_params(labelbottom=False)
    ax_j.plot(t, j[-1], color='black')

    ax_m.tick_params(labelbottom=True)
    ax_m.set_xlabel("$t~(\mathrm{ns})$")
    ax_m.set_xticks(t_ticks)
    ax_j.tick_params(labelbottom=True)
    ax_j.set_xlabel("$t~(\mathrm{ns})$")
    ax_j.set_xticks(t_ticks)
#    ax_j.set_ylabel("$j_e~(\mathrm{MA/cm^2})$")

#    fig.tight_layout()
#    plt.show()
    plt.savefig(save_dir+"fig2.pdf")
#    plt.savefig(save_dir+"fig2.png")
    plt.close()

if __name__ == '__main__':
    main()
