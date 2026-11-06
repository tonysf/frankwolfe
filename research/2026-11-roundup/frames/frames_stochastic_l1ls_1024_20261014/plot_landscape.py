"""A stated two-dimensional slice of the 1024-dimensional LS landscape."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FSTAR = 91/512


def slice_value(p, q, beta):
    # x[:64]=(1-p-q)/64; x[64:128]=-q/64; x[128:]=p/896.
    # All shown p,q>=0, p+q<=1, so ||x||_1=1 exactly.
    value = ((1+p+q)**2 + (1.5-q)**2)/128 + (p-7)**2/1792 + 1/8
    return value + q*q/(128*beta) - FSTAR


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    p, q = np.meshgrid(np.linspace(0, .5, 301), np.linspace(0, .35, 301))
    betas = [1., 32768**-.25]
    fig, axs = plt.subplots(1, 3, figsize=(14.8, 4.1), constrained_layout=True)
    for ax, beta in zip(axs[:2], betas):
        z = slice_value(p, q, beta)
        image = ax.contourf(p, q, z, levels=25, cmap='viridis')
        ax.contour(p, q, z, levels=12, colors='white', linewidths=.4, alpha=.5)
        qstar = beta/(2*(1+2*beta))
        ax.plot([0, .5], [0, 0], 'k-', lw=3, label='Original-feasible edge')
        ax.scatter([0], [qstar], marker='*', color='#ff7f0e', s=130, clip_on=False,
                   zorder=10, label='Exact smoothed solution')
        ax.scatter([0], [0], marker='o', color='black', s=35, clip_on=False,
                   zorder=10, label='Original solution')
        ax.set(xlabel='Mass on 896 small-target coordinates: $p$',
               ylabel='Negative mass on 64 coordinates: $q$',
               title=fr'$\Phi_\beta(x(p,q))-f^*$, $\beta={beta:.4g}$')
        fig.colorbar(image, ax=ax, shrink=.8, format='%.3f')
    axs[0].legend(fontsize=8, loc='upper left')
    qq = np.linspace(0, .35, 501)
    for beta, color in [(1, '#d95f02'), (.25, '#1b9e77'), (betas[-1], '#7570b3')]:
        axs[2].plot(qq, slice_value(0, qq, beta), color=color, label=fr'$\beta={beta:.4g}$')
        qstar = beta/(2*(1+2*beta))
        axs[2].scatter([qstar], [slice_value(0, qstar, beta)], color=color, s=25)
    axs[2].scatter([0], [0], color='black', zorder=10)
    axs[2].axhline(0, color='.6', lw=.7)
    axs[2].set(xlabel='Negative mass $q$ (slice $p=0$)',
               ylabel=r'$\Phi_\beta(x(0,q))-f^*$', title='Vanishing smoothing bias')
    axs[2].text(.03, .95, 'Original objective is $+\infty$ for $q>0$',
                va='top', transform=axs[2].transAxes, fontsize=9)
    axs[2].legend(fontsize=8, loc='center left')
    fig.savefig(OUT/'landscape.png', dpi=170)
    fig.savefig(OUT/'landscape.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
