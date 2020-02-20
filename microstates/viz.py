"""
Functions to visualize microstates.
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import mne

def plot_segmentation(segmentation, data, times):
    """Plot a microstate segmentation.

    Parameters
    ----------
    segmentation : list of int
        For each sample in time, the index of the state to which the sample has
        been assigned.
    times : list of float
        The time-stamp for each sample.
    """
    gfp = np.mean(data ** 2, axis=0)

    n_states = len(np.unique(segmentation))
    plt.figure(figsize=(6 * np.ptp(times), 2))
    cmap = plt.cm.get_cmap('plasma', n_states)
    plt.plot(times, gfp, color='black', linewidth=1)
    for state, color in zip(range(n_states), cmap.colors):
        plt.fill_between(times, gfp, color=color,
                         where=(segmentation == state))
    norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.title('Segmentation into %d microstates' % n_states)
    plt.autoscale(tight=True)
    plt.tight_layout()


def plot_maps(maps, info):
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_channels, n_maps)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    """
    plt.figure(figsize=(2 * len(maps), 2))
    for i, t_map in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        mne.viz.plot_topomap(t_map, pos=info)
        plt.title('%d' % i)
