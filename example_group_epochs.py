# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:04:32 2020

@author: Dragana
"""
import mne
import microstates as mst
import numpy as np

HC_RS_path = 'C:/Users/.../Documents/RS_EEG/'

subj_folder = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05']

# Parameteres setting up

chan_to_drop = ['E67',  'E73',  'E247', 'E251', 'E256', 'E243', 'E246', 'E250', 
                'E255', 'E82',  'E91',  'E254', 'E249', 'E245', 'E242', 'E253',
                'E252', 'E248', 'E244', 'E241', 'E92',  'E102', 'E103', 'E111', 
                'E112', 'E120', 'E121', 'E133', 'E134', 'E145', 'E146', 'E156', 
                'E165', 'E166', 'E174', 'E175', 'E187', 'E188', 'E199', 'E200', 
                'E208', 'E209', 'E216', 'E217', 'E228', 'E229', 'E232', 'E233',  
                'E236', 'E237', 'E240', 'E218', 'E227', 'E231', 'E235', 'E239', 
                'E219', 'E225', 'E226', 'E230', 'E234', 'E238']

pax = len(subj_folder) # number of participants
n_states = 4
n_inits = 1
EGI256 = True

if EGI256 == True:
    n_channels = 256 - len(chan_to_drop)

grouped_maps = np.array([], dtype=np.int64).reshape(0, n_channels)
for i, f in enumerate(subj_folder):
    fname = HC_RS_path + f + '/' + f +'_clean-epo.fif'
    epochs = mne.read_epochs(fname, preload=True)
    if EGI256 == True:
        epochs.drop_channels(chan_to_drop)
    data = epochs.get_data()
    # Segment the data in microstates
    maps, segmentation, gev, gfp_peaks = mst.segment(data, n_states, n_inits)
    grouped_maps = np.concatenate((grouped_maps, maps), axis=0)

# Transpose the maps from maps(n_maps, n_channels) to maps(n_channels, n_maps)
# and treat the n_maps as a sample in time. 
grouped_maps_T = grouped_maps.transpose()

# Find the group maps using k-means clustering
group_maps, _, group_gev = mst.groupsegment(grouped_maps_T, n_states, n_inits)

for i, subj in enumerate(subj_folder):
    fname = HC_RS_path + f + '/' + f +'_clean-epo.fif'
    epochs = mne.read_epochs(fname, preload=True)
    if EGI256 == True:
        epochs.drop_channels(chan_to_drop)
    data = epochs.get_data()
    # Compute final microstate segmentations on the original data
    activation = group_maps.dot(data)
    segmentation = np.argmax(activation ** 2, axis=0)

# Plot the maps
mst.viz.plot_maps(group_maps, epochs.info)