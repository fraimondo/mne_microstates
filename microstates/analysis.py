"""
Created on Wed Feb 19 13:36:13 2020

Part of the following functions are adapted from 
the toolbox eeg_microstates
https://github.com/Frederic-vW/eeg_microstates

Microstates analysis functions
"""
import numpy as np

def p_empirical(segmentation, n_epochs, n_samples, n_states=4, epoched_data=False):
    """Empirical symbol distribution
    Or in other words of Michel2018, Segment Count Density:
        the fraction of total recording time for which a given microstate
        is dominant. 

    Args:
        segmentation : ndarray, shape (n_samples,)
            For each sample, the index of the microstate to which the sample has
            been assigned.
        n_states : int
            The number of unique microstates to find. Defaults to 4.
        epoched_data : bool
            If True it means the segmentation was done on epoched data. 
            The transitions between the last microstate of epoch n and the first 
            microstate of epoch n+1, should not be taken into consideration. 
            Defaults to False.
        n_epochs : int
            The number of epochs of the segmented file.
        n_samples : int
            The number of samples in an epoch. 
        
    Returns:
        p  : ndarray, (n_states,)
            Empirical distribution
    """
    
    if epoched_data == True:
        all_p = []
        p = np.zeros(n_states)
        n = len(segmentation)
        for i in range(n_epochs):
            for j in range(n_samples):
                p[segmentation[(n_samples*i)+(j)]] += 1.0
            p /= n # dividing by n here and not by n_samples for value accuracy
            all_p.append(p)
        all_p = np.vstack(all_p)
        # sums the probabilities across all epochs
        all_p_sum = np.sum(all_p, axis=0)
        p = all_p_sum
        
    elif epoched_data == False:
        p = np.zeros(n_states)
        n = len(segmentation)
        for i in range(n):
            p[segmentation[i]] += 1.0
        p /= n
    
    return p

def mean_dur(segmentation, sfreq, n_states=4):
    """Mean duration of segments
        Average duration that a microstate remains stable. 

    Args:
        segmentation : ndarray, shape (n_samples,)
            For each sample, the index of the microstate to which the sample has
            been assigned.
        sfreq : sampling frequency 
        n_states: number of microstate clusters
    Returns:
        mean_durs : ndarray, shape (n_states,)
            the mean durations per state in seconds
    """
    durs = np.zeros((int(n_states), 2)) 
    n = len(segmentation)
    dur = 0
    for i in range(n):
        dur += 1
        if i == (n-1) or (segmentation[i+1] != segmentation[i]):
            durs[segmentation[i]][0] += dur
            durs[segmentation[i]][1] += 1
            dur = 0 
    mean_durs = durs[:, 0] / durs[:, 1]
    mean_durs /= sfreq
    return mean_durs

def T_empirical(segmentation, n_states):
    """Empirical transition matrix

    Args:
        segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
        n_states: number of microstate clusters
    Returns:
        T: empirical transition matrix
    """
    T = np.zeros((n_states, n_states))
    n = len(segmentation)
    for i in range(n-1):
        T[segmentation[i], segmentation[i+1]] += 1.0
    p_row = np.sum(T, axis=1)
    for i in range(n_states):
        if ( p_row[i] != 0.0 ):
            for j in range(n_states):
                T[i,j] /= p_row[i]  # normalize row sums to 1.0
    return T


def print_matrix(T):
    """Console-friendly output of the matrix T.

    Args:
        T: matrix to print
    """
    for i, j in np.ndindex(T.shape):
        if (j == 0):
            print("\t\t|{:.3f}".format(T[i,j]), end=' ')
        elif (j == T.shape[1]-1):
            print("{:.3f}|\n".format(T[i,j]), end=' ')
        else:
            print("{:.3f}".format(T[i,j]), end=' ')
