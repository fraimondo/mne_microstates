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
n_states = 4
n_inits = 5
EGI256 = True 
sfreq = epochs.info['sfreq']

# Removing channels 
if EGI256 == True:
    epochs.drop_channels(['E67', 'E73', 'E247', 'E251', 'E256', 'E243', 'E246', 'E250', 
                          'E255', 'E82', 'E91', 'E254', 'E249', 'E245', 'E242', 'E253',
                          'E252', 'E248', 'E244', 'E241', 'E92', 'E102', 'E103', 'E111', 
                          'E112', 'E120', 'E121', 'E133', 'E134', '145', 'E146', 'E156', 
                          'E165', 'E166', 'E174', 'E175', 'E187', 'E188', 'E199', 'E200', 
                          'E208', 'E209', 'E216', 'E217', 'E228', 'E229', 'E232', 'E233',  
                          'E236', 'E237', 'E240', 'E218', 'E227', 'E231', 'E235', 'E239', 
                          'E219', 'E225', 'E226', 'E230', 'E234', 'E238'])

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
print("\n\t\tGlobal explained variance (GEV):")
print ("\t\t" + str(gev))

# Mean durations of states 
mean_durs = mst.analysis.mean_dur(segmentation, epochs.info['sfreq'], n_states)
print("\n\t\t Mean microstate durations in ms:\n")
for i in range(n_states): 
    print("\t\tp_{:d} = {:.3f}".format(i, mean_durs[i]*1000))
