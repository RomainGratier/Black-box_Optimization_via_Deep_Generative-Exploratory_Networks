import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from matplotlib import ticker
import seaborn as sns
sns.set()

from morphomnist import io


# def plot_radial_hist(angles, ax=None, **kwargs):
#     if ax is None:
#         ax = plt.gca()
#     lims = [-np.pi / 4, np.pi / 4]
#     bin_width = np.pi / 60
#     bins = np.arange(lims[0], lims[1] + bin_width, bin_width) - bin_width / 2
#     counts, bins, patches = ax.hist(angles, bins=bins, **kwargs)
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)
#     ax.set_thetalim(lims)
#     ax.set_rorigin(-ax.get_rmax())
#     ax.set_xticks(np.arange(lims[0] + 1e-3, lims[1] + 2e-3, np.pi / 12))
#     colour_hist(patches, counts, plt.cm.Blues)


# def colour_hist(bars, counts, cmap):
#     for count, bar in zip(counts, bars):
#         bar.set_facecolor(cmap(count / max(counts)))

def violine_plots(data_path, output_path, cols=['thickness','slant', 'width', 'height']):
    
    df = pd.read_csv(os.path.join(data_path, "train-morpho.csv"), index_col='index')
    df['digit'] = io.load_idx(os.path.join(data_path, "train-labels-idx1-ubyte.gz"))
    #df['slant'] = np.rad2deg(np.arctan(-df['shear']))

    labels = cols
    fig, axs = plt.subplots(2, len(cols), sharex='col', sharey='row', figsize=(12, 4),
                            gridspec_kw=dict(height_ratios=[10, 1], hspace=.1, wspace=.1, left=0,
                                             right=1))

    def format_violinplot(parts):
        for pc in parts['bodies']:
            pc.set_facecolor('#1f77b480')
            pc.set_edgecolor('C0')
            pc.set_alpha(None)
    
    for c, col in enumerate(cols):
        ax = axs[0, c]
        parts = ax.violinplot([df.loc[df['digit'] == d, col].values.tolist() for d in range(10)], positions=np.arange(10), vert=False, widths=0.7, showextrema=False, showmedians=True)
        format_violinplot(parts)
        format_violinplot(axs[1, c].violinplot(df[col], vert=False, widths=.7, showextrema=False, showmedians=True))
        ax.set_title(labels[c])
        ax.set_axisbelow(True)
        ax.grid(axis='x')
        axs[1, c].set_axisbelow(True)
        axs[1, c].grid(axis='x')

    axs[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[0, 0].set_ylabel("Digit")
    axs[1, 0].set_yticks([1])
    axs[1, 0].set_yticklabels(["All"])
    axs[1, 0].set_ylim(.5, 1.5)
    #for ax in axs[:, 1]:
    #    ax.axvline(0., lw=1., ls=':', c='k')
    #axs[1, 1].set_xlim(-46, 46)
    #axs[1, 1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}$\degree$"))
    plt.savefig(os.path.join(output_path, data_path.split('/')[-1]+'.pdf' ), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    ROOT = "../data/" 
    OUPUT_PTH = "../fig/"
    paths = [os.path.join(ROOT, "processed/original_thic_resample"), os.path.join(ROOT, "morpho_mnist/original"), os.path.join(ROOT, "morpho_mnist/thic")]
    
    for path in paths:
        violine_plots(path, OUPUT_PTH)