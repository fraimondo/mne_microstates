"""
Created on Wed Feb 19 13:36:13 2020

Part of the following functions are adapted from 
the toolbox eeg_microstates
https://github.com/Frederic-vW/eeg_microstates

Microstates analysis functions
"""
import numpy as np
from scipy.stats import chi2 

def p_empirical(segmentation, n_epochs, n_samples, n_states=4, epoched_data=False):
    """Empirical symbol distribution
    Or in other words of Michel2018, Segment Count Density:
        the fraction of total recording time for which a given microstate
        is dominant. 

    Args:
        segmentation : ndarray, shape (n_samples,)
            For each sample, the index of the microstate to which the sample has
            been assigned.
        n_epochs : int
            The number of epochs of the segmented file.
        n_samples : int
            The number of samples in an epoch.
        n_states : int
            The number of unique microstates to find. Defaults to 4.
#        epoched_data : bool
#            If True it means the segmentation was done on epoched data. 
#            The transitions between the last microstate of epoch n and the first 
#            microstate of epoch n+1, should not be taken into consideration. 
#            Defaults to False.
        
    Returns:
        p  : ndarray, (n_states,)
            Empirical distribution
    """
    
#    if epoched_data == True:
#        all_p = []
#        # Array with the count of mst occurences
#        p = np.zeros(n_states) # instead: n_states+1
#        n = len(segmentation)
#        for i in range(n_epochs):
#            for j in range(n_samples):
#                # if segmentation[(n_samples*i)+(j)] != 88: 
#                p[segmentation[(n_samples*i)+(j)]] += 1.0
#            p /= n # dividing by n here and not by n_samples for value accuracy
#            all_p.append(p)
#        all_p = np.vstack(all_p)
#        # sums the probabilities across all epochs
#        all_p_sum = np.sum(all_p, axis=0)
#        p = all_p_sum
        
    if epoched_data == False:
        p = np.zeros(n_states)
        n = len(segmentation)
        for i in range(n):
            p[segmentation[i]] += 1.0
        p /= n
    
    return p

def mean_dur(segmentation, sfreq, n_states=5):
    """Mean duration of segments
        Average duration that a microstate remains stable. 

    Args:
        segmentation : ndarray, shape (n_samples,)
            For each sample, the index of the microstate to which the sample has
            been assigned.
        sfreq : float
            Sampling frequency of the signal. 
        n_states : int
            The number of unique microstates to find. Defaults to 4.
    Returns:
        mean_durs : ndarray, shape (n_states,)
            the mean durations per state in seconds
        all_durs : list of lists
            Contains all the durations of each microstate. 
    """
    durs = np.zeros((int(n_states), 2)) 
    n = len(segmentation)
    dur = 0
    all_durs = [[] for i in range(n_states)]
    for i in range(n):
        dur += 1
        if i == (n-1) or (segmentation[i+1] != segmentation[i]):
            durs[segmentation[i]][0] += dur
            durs[segmentation[i]][1] += 1
            all_durs[segmentation[i]].append(dur)
            dur = 0
    mean_durs = durs[:, 0] / durs[:, 1]
    mean_durs /= sfreq # we get the mean_durs in seconds
    return mean_durs, all_durs

def T_empirical(segmentation, n_epochs, epoched_data=True, n_states=4):
    """Empirical transition matrix
    The transition is from row to column.
    Eg. [0,0] (mst1) --> [0,1] (mst2)
    
    Args:
        segmentation : ndarray, shape (n_samples,)
            For each sample, the index of the microstate to which the sample has
            been assigned.
        n_states : int
            The number of unique microstates to find. Defaults to 4.
    Returns:
        T: empirical transition matrix
    """
    
    if epoched_data == False:
        # for a single epoch
        T = np.zeros((n_states, n_states))
        n = len(segmentation)
        for i in range(n-1):
             T[segmentation[i], segmentation[i+1]] += 1.0
        p_row = np.sum(T, axis=1)
        for i in range(n_states):
            if ( p_row[i] != 0.0 ):
                for j in range(n_states):
                    T[i,j] /= p_row[i]  # normalize row sums to 1.0
    
    if epoched_data == True:  
        # for multiple epochs          
        # For data in epochs: columns:epochs, rows:segmentations 
        T = np.zeros((n_states, n_states, n_epochs))
        n = len(segmentation)    
        for epo in range(n_epochs):
            for i in range(n-1):
                T[segmentation[i,epo], segmentation[i+1,epo], epo] += 1.0
        p_row = np.sum(T, axis=1) # 2D (row_sums, epochs)
        for epo in range(n_epochs):
            for i in range(n_states):
                if ( p_row[i,epo] != 0.0 ):
                    for j in range(n_states):
                        T[i,j,epo] /= p_row[i,epo]  # normalize row sums to 1.0
        
        # Average all the transition matrixes accross the different epochs            
        T = np.average(T, axis=2)
    
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
            

def symmetryTest(X, ns, alpha, verbose=True):
    """Test symmetry of the transition matrix of symbolic sequence X with
    ns symbols
    cf. Kullback, Technometrics (1962)
    H0 -> the transition probabilities matrix is symmetric
    H1 -> the transition probabilities matrix is asymmetric
    Args:
        x  --> segmentation : symbolic sequence, symbols = [0, 1, 2, ...]
        ns --> n_states : number of symbols
        alpha: significance level  0.01
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\n\t\tSYMMETRY:" )
    n = len(X)
    f_ij = np.zeros((ns,ns))
    for t in range(n-1):
        i = X[t]
        j = X[t+1]
        f_ij[i,j] += 1.0
    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i,j]*f_ij[j,i]
            if (f > 0):
                T += (f_ij[i,j]*np.log((2.*f_ij[i,j])/(f_ij[i,j]+f_ij[j,i])))
    T *= 2.0
    df = ns*(ns-1)/2
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df)))
    return p, T, df