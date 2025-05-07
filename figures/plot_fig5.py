import os
import toml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm


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

    t = np.loadtxt(directory+'t.txt')
    t = np.concatenate(([-dt], t))

    import glob
    files = sorted(glob.glob(directory+'*.txt'))
    exclude_files = {'j.txt', 't.txt'}
    files = [f for f in files if os.path.basename(f) not in exclude_files]

    m = []
    for file in files:
        tmp_m = np.loadtxt(file)
        tmp_m = np.concatenate(([m0], tmp_m))
        m.append(tmp_m)
    m = np.array(m)

    j = np.loadtxt(directory+"j.txt")
    j = np.concatenate(([0e0], j))

    return t, m, j

def main():
    save_dir = "./output/"
    os.makedirs(save_dir, exist_ok=True)

    currents = ["05.0", "10.0"]
    size = "100x50x1"
    j_ticks = [0, 5, 10]
    t_ticks = [0, 0.5, 1.0, 1.5]

#    currents = ["8e+10", "9e+10", "1e+11", "2e+11"]
#    size = "80x25x1"
#    j_ticks = [0, 5, 10, 15, 20]
#    t_ticks = [0, 0.5, 1.0, 1.5, 2.0]

    fig = plt.figure(figsize=(9, 5))
    gs = gridspec.GridSpec(3, len(currents), figure=fig, wspace=0.11, hspace=0.2, left=0.1, right=0.98, top=0.95, bottom=0.13)

    m_ticks = [-1, -0.5, 0, 0.5, 1]

    labels = ['$m_x$', '$m_y$', '$m_z$']
    colors = ['tab:red', 'tab:blue', 'tab:green']
    lss = ['-', ':', '-.']

    for i, current in enumerate(currents):
        load_dir = "../data/thermal/M750/J"+current+"e10_T300/"

        t, m, j = load(load_dir)

        ax_j = fig.add_subplot(gs[0, i])
        ax_j.set_xlim([-0.01,t_ticks[-1]])
        ax_j.set_xticks(t_ticks)
        ax_j.xaxis.set_minor_locator(AutoMinorLocator())
        ax_j.yaxis.set_minor_locator(AutoMinorLocator())
        ax_j.tick_params(labelbottom=False)
        ax_j.set_ylim(-0.1, j_ticks[-1]*1.1)
        ax_j.set_yticks(j_ticks)
        ax_j.set_ylabel(r"$j_e~(\mathrm{MA/cm^2})$", labelpad=17)
        ax_j.plot(t, j, color='black')

        ax_m = fig.add_subplot(gs[1:3, i])
        ax_m.set_xlabel("$t~(\mathrm{ns})$")
        ax_m.set_xlim([-0.01, t_ticks[-1]])
        ax_m.xaxis.set_minor_locator(AutoMinorLocator())
        ax_m.yaxis.set_minor_locator(AutoMinorLocator())
        ax_m.set_xticks(t_ticks)
        ax_m.set_ylim([-1,1])
        ax_m.set_ylabel(r"$\boldsymbol{m}$")
        ax_m.set_yticks(m_ticks)

##        n = 2
##        selection = np.random.choice(len(m), size=n, replace=False)
##        cmap = mpl.colormaps['coolwarm']
##        colors = [cmap(k / (n - 1)) for k in range(n)]
#
        n = 1
        selection = np.random.choice(len(m), size=n, replace=False)

        for k, (label, color, ls) in enumerate(zip(labels, colors, lss)):
            data = m[:,:,k]
            mean = np.mean(data, axis=0)
            cov  = np.cov(data, rowvar=False)
            std  = np.sqrt(np.diag(cov))

            # 平均 + 信頼区間
#            plt.plot(t, m, color=color, label=label)
            plt.fill_between(t,
#                             mean - 1.96 * std, # 95%
#                             mean + 1.96 * std, # 95%
                             mean - 2.576 * std, # 99%
                             mean + 2.576 * std, # 99%
#                             mean - 3.291 * std, # 99.9%
#                             mean + 3.291 * std, # 99.9%
                             color=color, alpha=0.2, label='95% CI')


            for l, selected in enumerate(selection):
                ax_m.plot(t, m[selected][:,k], label=label, color=color, ls=ls)
#                ax_m.plot(t, m[selected][:,k], label=label, color=colors[l])

        if i > 0:
            ax_j.set_ylabel("")
            ax_j.tick_params(labelleft=False)
            ax_m.set_ylabel("")
            ax_m.tick_params(labelleft=False)

        if i == len(currents)-1:
            ax_j.text(0.7, 1.02, r"$T = 300~\mathrm{K}$", transform=ax_j.transAxes)

            t_x = 1200
            ax_m.annotate(
                "$m_x$",
                xy=(t[t_x], -0.95),
                xytext=(t[t_x], -0.75),
                arrowprops=dict(
                    arrowstyle='->',                    # ← 矢印ヘッドなし！ただの線
                    color='tab:red',
                    lw=1,
                    mutation_scale=10
                ),
                ha='center',
                color="tab:red"
            )
            t_y = 650
            ax_m.annotate(
                "$m_y$",
                xy=(t[t_y], 0.45),
                xytext=(t[t_y], 0.75),
                arrowprops=dict(
                    arrowstyle='->',                    # ← 矢印ヘッドなし！ただの線
                    color='tab:blue',
                    lw=1,
                    mutation_scale=10
                ),
                ha='center',
                color="tab:blue"
            )
            t_z = 250
            ax_m.annotate(
                "$m_z$",
                xy=(t[t_z], 0.02),
                xytext=(t[t_z], -0.6),
                arrowprops=dict(
                    arrowstyle='->',                    # ← 矢印ヘッドなし！ただの線
                    color='tab:green',
                    lw=1,
                    mutation_scale=10
                ),
                ha='center',
                color="tab:green"
            )


#        ax_m.text(0.1, 0.05, f"{episode}th", transform=ax_m.transAxes)

#    plt.show()
    plt.savefig(save_dir+"fig5.pdf")
    plt.close()

if __name__ == '__main__':
    main()
