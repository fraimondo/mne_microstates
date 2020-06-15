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
        
    Note: T_empirical from the same epoched and continuous (stacked epoched) data 
          differ in the 3rd/4th decimal place. 
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
        # Average all the transition matrixes accross the different epochs            
        T = np.average(T, axis=2)
        p_row = np.sum(T, axis=1) # 2D (row_sums, epochs)
        for i in range(n_states):
            if (p_row[i] != 0.0):
                for j in range(n_states):
                    T[i,j] = T[i,j] / p_row[i] # normalize row sums to 1.0
    return T


#def T_theoretical(p_hat, n_states):
#    T_th = np.array([[p_hat[i]*p_hat[j] for i in range(n_states)] for j in range(n_states)])  
#    return T_th

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
            

def symmetryTest(X, ns=4, alpha=0.01, verbose=True):
    """Test symmetry of the transition matrix of symbolic sequence X with
    ns symbols
    cf. Kullback, Technometrics (1962)
    H0 -> the transition probabilities matrix is symmetric
    H1 -> the transition probabilities matrix is asymmetric
    Args:
        x  --> segmentation : symbolic sequence, symbols = [0, 1, 2, ...]
            Should be the  continuous segmentation: seg_smooth_cont
            
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

def testMarkov0(seg, n_epochs, alpha, ns, epoched_data=True, verbose=True):
    """Test zero-order Markovianity of symbolic sequence x with ns symbols.
    Null hypothesis: zero-order MC (iid) <=>
    p(X[t]), p(X[t+1]) independent
    cf. Kullback, Technometrics (1962)
    
    Args:
        seg: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """
    if verbose:
        print( "\t\tZERO-ORDER MARKOVIANITY:" )
    n = len(seg)
    
    T=0
    if epoched_data == False:
        # for a single epoch
        f_ij = np.zeros((ns,ns))
        f_i = np.zeros(ns)
        f_j = np.zeros(ns)
        # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
        for t in range(n-1):
            i = seg[t]
            j = seg[t+1]
            f_ij[i,j] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0
        T = 0.0 # statistic
        for i, j in np.ndindex(f_ij.shape):
            f = f_ij[i,j]*f_i[i]*f_j[j]
            if (f > 0):
                T += (f_ij[i,j] * np.log((n*f_ij[i,j])/(f_i[i]*f_j[j])))
        T *= 2.0
        
    if epoched_data == True:  ### not sure if this can be done like this
        # for multiple epochs          
        # For data in epochs: columns:epochs, rows:segmentations 
        # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
        all_T = []
        for epo in range(n_epochs):
            T = 0.0 # statistic
            f_ij = np.zeros((ns,ns))
            f_i = np.zeros(ns)
            f_j = np.zeros(ns)
            seg_epo = seg[:, epo]
            for t in range(n-1):
                i = seg_epo[t]
                j = seg_epo[t+1]
                f_ij[i,j] += 1.0
                f_i[i] += 1.0
                f_j[j] += 1.0
            for i, j in np.ndindex(f_ij.shape):
                f = f_ij[i,j]*f_i[i]*f_j[j]
                if (f > 0):
                    T += (f_ij[i,j] * np.log((n*f_ij[i,j])/(f_i[i]*f_j[j])))
            T *= 2.0
            all_T.append(T)
        all_T_ave = np.average(all_T)
        T = all_T_ave
        
    # Degrees of Freedom:
    df = (ns-1.0) * (ns-1.0)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    print(("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df)))
    # For ns = 4, len(x)=201, alpha=0.01
        # Marginal value of T for significance is 21.68. For T > 21.68, p < alpha
    return p, T, df


def testMarkov1(X, ns, alpha, verbose=True):
    """Test first-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\n\t\tFIRST-ORDER MARKOVIANITY:" )
    n = len(X)
    f_ijk = np.zeros((ns,ns,ns))
    f_ij = np.zeros((ns,ns))
    f_jk = np.zeros((ns,ns))
    f_j = np.zeros(ns)
    for t in range(n-2):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        f_ijk[i,j,k] += 1.0
        f_ij[i,j] += 1.0
        f_jk[j,k] += 1.0
        f_j[j] += 1.0
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i][j][k]*f_j[j]*f_ij[i][j]*f_jk[j][k]
        if (f > 0):
            T += (f_ijk[i,j,k]*np.log((f_ijk[i,j,k]*f_j[j])/(f_ij[i,j]*f_jk[j,k])))
    T *= 2.0
    df = ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df)))
    return p, T, df


def testMarkov2(X, ns, alpha, verbose=True):
    """Test second-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t], X[t-1]) = p(X[t+1] | X[t], X[t-1], X[t-2])
    cf. Kullback, Technometrics (1962), Table 10.2.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\n\t\tSECOND-ORDER MARKOVIANITY:" )
    n = len(X)
    f_ijkl = np.zeros((ns,ns,ns,ns))
    f_ijk = np.zeros((ns,ns,ns))
    f_jkl = np.zeros((ns,ns,ns))
    f_jk = np.zeros((ns,ns))
    for t in range(n-3):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        l = X[t+3]
        f_ijkl[i,j,k,l] += 1.0
        f_ijk[i,j,k] += 1.0
        f_jkl[j,k,l] += 1.0
        f_jk[j,k] += 1.0
    T = 0.0
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        f = f_ijkl[i,j,k,l]*f_ijk[i,j,k]*f_jkl[j,k,l]*f_jk[j,k]
        if (f > 0):
            T += (f_ijkl[i,j,k,l]*np.log((f_ijkl[i,j,k,l]*f_jk[j,k])/(f_ijk[i,j,k]*f_jkl[j,k,l])))
    T *= 2.0
    df = ns*ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df)))
    return p, T, df