"""
Created on Wed Feb 19 13:36:13 2020

Part of the following functions are adapted from 
the toolbox eeg_microstates
https://github.com/Frederic-vW/eeg_microstates

Microstates analysis functions
"""
import numpy as np

def p_empirical(segmentation, n_states=4):
    """Empirical symbol distribution

    Args:
        segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
        n_states: number of microstate clusters
    Returns:
        p: empirical distribution
    """
    p = np.zeros(n_states)
    n = len(segmentation)
    for i in range(n):
        p[segmentation[i]] += 1.0
    p /= n
    return p


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
