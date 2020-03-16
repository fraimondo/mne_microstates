"""
Functions to segment EEG into microstates. Based on the Microsegment toolbox
for EEGlab, written by Andreas Trier Poulsen [1]_.

References
----------
.. [1]  Poulsen, A. T., Pedroni, A., Langer, N., &  Hansen, L. K. (2018).
        Microstate EEGlab toolbox: An introductionary guide. bioRxiv.
"""
import warnings
import numpy as np
from scipy.stats import zscore
from scipy.signal import find_peaks
from scipy.linalg import eigh

import mne
from mne.utils import logger, verbose


@verbose
def segment(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
            use_peaks=True, normalize=False, min_peak_dist=2, max_n_peaks=10000,
            random_state=None, verbose=None):
    """Segment a continuous signal into microstates.

    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithm
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in
    n_states : int
        The number of unique microstates to find. Defaults to 4.
    n_inits : int
        The number of random initializations to use for the k-means algorithm.
        The best fitting segmentation across all initializations is used.
        Defaults to 10.
    max_iter : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    thresh : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    normalize : bool
        Whether to normalize (z-score) the data across time before running the
        k-means algorithm. Defaults to ``False``.
    min_peak_dist : int
        Minimum distance (in samples) between peaks in the GFP. Defaults to 2.
    max_n_peaks : int | None
        Maximum number of GFP peaks to use in the k-means algorithm. Chosen
        randomly. Set to ``None`` to use all data. Defaults to 10000.
    random_state : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.
    verbose : int | bool | None
        Controls the verbosity.
    use_peaks : bool
        Whether to find the GFP peaks or not. True if finding maps per subject,
        False if finding maps for a group of subjects - in this case we don't 
        need the segmentation.  

    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    best_gev : the best Global Explained Variance (GEV)
    peaks : the GFP peaks 

    References
    ----------
    .. [1] Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).
           Segmentation of brain electrical activity into microstates: model
           estimation and validation. IEEE Transactions on Biomedical
           Engineering.
    """
    logger.info('Finding %d microstates, using %d random intitializations' %
                (n_states, n_inits))
    
    if normalize:
        data = zscore(data, axis=1)

    if use_peaks == True: 
        if len(data.shape) == 3:
            epo_data = True
            logger.info('Finding microstates from epoched data.')
            n_epochs, n_chans, n_samples = data.shape
            # Make 2D and keep events info
            data = np.hstack(data)
    #        events = np.arange(0, data.shape[1], n_samples)
        else:
            epo_data = False

    # Find peaks in the global field power (GFP)
    gfp = np.mean(data ** 2, axis=0)    
    
    if use_peaks == True: 
        # Find the peaks in the GFP
        peaks, _ = find_peaks(gfp, distance=min_peak_dist)

        ######################################################################
        # Add a part that removed the lower 15% of peaks.    
    
        ######################################################################
    
        n_peaks = len(peaks)
    
        # Limit the number of peaks by randomly selecting them
        if max_n_peaks is not None:
            max_n_peaks = min(n_peaks, max_n_peaks)
            if not isinstance(random_state, np.random.RandomState):
                random_state = np.random.RandomState(random_state)
            chosen_peaks = random_state.choice(n_peaks, size=max_n_peaks,
                                               replace=False)
            peaks = peaks[chosen_peaks]
        
        # Taking the data only at the GFP peaks and recalculating the GFP    
        data_peaks = data[:, peaks]
        gfp = np.mean(data ** 2, axis=0)
        # Shouldn't it be this?
        #        gfp_hm = np.std(data, axis=0)

        # Cache this value for later
        gfp_sum_sq = np.sum(gfp ** 2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_maps = None
    best_segmentation = None
    
    if use_peaks == False:
        data_peaks = data

    for _ in range(n_inits):
        maps = _mod_kmeans(data_peaks, n_states, n_inits, max_iter, 
                           thresh, random_state, verbose)
        
        activation = maps.dot(data)
        segmentation = np.argmax(activation ** 2, axis=0)
        ######################################################################
        # Here add a part in which you select the microstates around a border, 
        # and you remove those indices in the segmentation
        # but also in the data. 
        
        # Or maybe not, maybe we can just remove them before doing the analyses.
        # The way p_empirical() is calculated now for the epoched data is alligned
        # with this. 
        
        ######################################################################
        map_corr = _corr_vectors(data, maps[segmentation].T)
#        limmap = [i for i in map_corr if i > 0.5 or i < -0.5]

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
        logger.info('GEV of found microstates: %f' % gev)
        if gev > best_gev:
            best_gev, best_maps, best_segmentation = gev, maps, segmentation
    
    if use_peaks == True:
        return best_maps, best_segmentation, best_gev, peaks #, map_corr, limmap
    elif use_peaks == False:
        return best_maps, best_gev
    
def mark_border_msts(segmentation, n_epochs, n_samples, n_states=4):
    """ Marks the microstates surrounding the epoch edges.
    This is done because when we use epoched data, we cannot know when a 
    microstate would have ended or would have begun if we hadn't cut the data. 
    
    The samples of the segmentation which should not be used are attributed 
    the value 88. 
    
    
        Parameters
    ----------
    segmetation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    n_epochs : int
        The number of epochs of the segmented file.
    n_samples : int
        The number of samples in an epoch.
    n_states : int
        The number of unique microstates to find. Defaults to 4.
 
    Returns
    -------
    new_seg : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    
    """
    seg_new = segmentation
    
    for i in range(n_epochs):
                
        for j in range(n_samples):
            if j == 0:
                first_mst = seg_new[n_samples*i]
                seg_new[n_samples*i] = 88
            else:
                if seg_new[(n_samples*i)+j] == first_mst:
                    seg_new[(n_samples*i)+j] = 88
                else:
                    break 
                
        for j in range(n_samples):
            if j == 0:
                last_mst = seg_new[n_samples*(i+1) - 1]
                seg_new[n_samples*(i+1) - 1] = 88
            else:
                if seg_new[n_samples*(i+1) - (j+1)] == last_mst:
                    seg_new[n_samples*(i+1) - (j+1)] = 88
                else:
                    break
    return seg_new

@verbose
def _mod_kmeans(data_peaks, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                random_state=None, verbose=None):
    """The modified K-means clustering algorithm.

    See :func:`segment` for the meaning of the parameters and return
    values.
    
    Parameters
    ----------
    data_peaks : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in - selected from the original data
            at the chosen_peaks. 
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data_peaks.shape

    # Cache this value for later
    data_sum_sq = np.sum(data_peaks ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data_peaks[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data_peaks)
        segmentation = np.argmax(np.abs(activation), axis=0)
        # assigned_activations = np.choose(segmentations, all_activations)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                maps[state] = 0
                continue
           
            # Find largest eigenvector
            # cov = data[:, idx].dot(data[:, idx].T)
            # _, vec = eigh(cov, eigvals=(n_channels - 1, n_channels - 1))
            # maps[state] = vec.ravel()
            maps[state] = data_peaks[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data_peaks, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logger.info('Converged at %d iterations.' % iteration)
            break

        prev_residual = residual
    else:
        warnings.warn('Modified K-means algorithm failed to converge.')

    return maps


def _corr_vectors(A, B, axis=0):
    """Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).

    Parameters
    ----------
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.

    Returns
    -------
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)