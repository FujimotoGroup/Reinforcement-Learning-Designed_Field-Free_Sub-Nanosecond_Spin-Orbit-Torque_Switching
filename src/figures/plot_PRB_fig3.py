import toml
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

import glob

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

    load_dir = "../../data/finiteT/100x50x1/aG0.010/M750/*"
    j_ticks = [0, 5, 10]

    dirs = sorted(glob.glob(load_dir))

    label_b = 'for each episode'
    window = 10
    label_ma = f'moving average'

    jes = []
    historys = []
    moving_avgs = []
    for d in dirs:
        print(d)
        m = []
        config = toml.load(d+"/config.toml")
        j_e = str(config['simulation']['current'])

        history = np.loadtxt(d+'/reward_history.txt')
        episodes = np.arange(len(history))

        weights = np.ones(window) / window
        moving_avg = np.convolve(history, weights, mode='valid')

        jes.append(j_e)
        historys.append(history)
        moving_avgs.append(moving_avg)

    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(3, 1, figure=fig, wspace=0.2, hspace=0.2, left=0.02, right=0.96, top=0.96, bottom=0.08)

    axes = []
    for k, (j_e, history, moving_avg) in enumerate(zip(jes, historys, moving_avgs)):
        ax_history = fig.add_subplot(gs[k])
        ax_history.set_xlim([0, len(history)])
        ax_history.xaxis.set_minor_locator(AutoMinorLocator())
        ax_history.set_yticks([])
        ax_history.tick_params(labelbottom=False)
        ax_history.plot(episodes, history, color="tab:gray")
        ax_history.plot(moving_avg, color="black")
        axes.append(ax_history)
        ax_history.text(0.68, 0.03, r"$j_e = "+j_e+r"~\mathrm{MA/cm^2}$", transform=ax_history.transAxes)

        ax_history.axvspan(1950, 2000, color='yellow', alpha=0.5)

    axes[0].set_title("Rewards (a.u.)")
    axes[-1].set_xlabel("episode")
    axes[-1].tick_params(labelbottom=True)

    k = 1300+np.argsort(historys[0][1300:1500])[0]
    print(k)
    axes[0].annotate(
        label_b,
        xy=(k, historys[0][k]),
        xytext=(k, historys[0][k]-13),
        arrowprops=dict(arrowstyle='->', color='tab:gray', lw=1),
        ha='center',
        color="tab:gray"
    )
    k = 500+np.argsort(moving_avgs[0][500:1000])[0]
    axes[0].annotate(
        label_ma,
        xy=(k, moving_avgs[0][k]),
        xytext=(k, moving_avgs[0][k]-25),
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
        ha='center',
        color="black"
    )

#    fig.tight_layout()
#    plt.show()
    plt.savefig(save_dir+"PRB_fig3.pdf")
#    plt.savefig(save_dir+"PRB_fig3.png")
    plt.close()

if __name__ == '__main__':
    main()

