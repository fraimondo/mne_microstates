import mne
import sys

import microstates as mst


from mne.datasets import sample
fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)

raw.info['bads'] = ['MEG 2443', 'EEG 053']
raw.set_eeg_reference('average')
raw.pick_types(meg='mag', eeg=True, eog=True, ecg=True, stim=True)
raw.filter(1, 40)

# Clean EOG with ICA
ica = mne.preprocessing.ICA(0.99).fit(raw)
bads_eog, _ = ica.find_bads_eog(raw)
bads_ecg, _ = ica.find_bads_ecg(raw)
ica.exclude = bads_eog[:2] + bads_ecg[:2]
raw = ica.apply(raw)

event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=.5,
                    preload=True)

# Select sensor type
# raw.pick_types(meg=False, eeg=True)
epochs.pick_types(meg=False, eeg=True)


#================ Microstates ================#
# Parameteres setting up
n_states = 5
n_inits = 5

# Segment the data in 6 microstates
maps, segmentation, gev, gfp_peaks = mst.segment(
    epochs.get_data(), n_states, n_inits)

# Plot the topographic maps of the microstates and the segmentation
mst.viz.plot_maps(maps, epochs.info)
mst.viz.plot_segmentation(
    segmentation[:500], raw.get_data()[:, :500], raw.times[:500])


#================ Analyses ================#
# p_empirical 
p_hat = mst.analysis.p_empirical(segmentation, n_states)
print("\n\t\tEmpirical symbol distribution (RTT):\n")
for i in range(n_states): 
    print("\t\tp_{:d} = {:.3f}".format(i, p_hat[i]))

# T_empirical
T_hat = mst.analysis.T_empirical(segmentation, n_states)
print("\n\t\tEmpirical transition matrix:\n")
mst.analysis.print_matrix(T_hat)

# Peaks Per Second (PPS)
fs = epochs.info['sfreq']
pps = len(gfp_peaks) / (len(segmentation)/fs)  # peaks per second
print("\n\t\tGFP peaks per sec.: {:.2f}".format(pps))

# Global Explained Variance (GEV)
print("\n\t\tGlobal explained variance (GEV) per map:")
print ("\t\t" + str(gev))
print("\n\t\ttotal GEV: {:.2f}".format(gev.sum()))
