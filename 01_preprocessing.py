'''

Stage 1: Pre-processing

This script performs pre-processing on the EEG data. The pipeline
consists of bandpass filtering, notch filtering to remove line noise,
bad channel rejection with RANSAC, and ICA denoising. Cleaned
raw and epochs are saved in the MNE bids derivative folder.

'''

### Import Packages ###
import mne
import numpy as np
import pandas as pd
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)

import sys

from pyprep.find_noisy_channels import NoisyChannels

### Define Preprocessing Pipeline ###

def main(subject):
    print('Starting Pre-processing...')

    ### Load Data ###
    bids_root = 'inputs/data'
    sessions = get_entity_vals(bids_root, "session", ignore_sessions="on")
    datatype = "eeg"
    extensions = [".fdt", ".tsv"]  # ignore .json files
    bids_path = BIDSPath(root=bids_root, datatype=datatype)
    task = "N170"
    suffix = "eeg"
    subject = subject.zill(3)
    bids_path = bids_path.update(subject=subject, task=task, suffix=suffix)
    raw = read_raw_bids(bids_path=bids_path, verbose=False, extra_params = {'preload' : True})
    raw.rename_channels({'FP1' : 'Fp1','FP2' : 'Fp2'})
    raw.set_montage('standard_1020')

    ### Create Initial MNE Report ###
    report = mne.Report(title=f"Pre-processing Report: sub-{subject}")
    report.add_raw(raw=raw, title="Raw", psd=True)

    ### Bandpass Filter ###
    raw_bandpass = raw.copy().filter(0.25,125)
    report.add_raw(raw=raw_bandpass, title="Bandpass Filter", psd=True)

    ### Notch Filter Powerline Frequency & Harmonics ###
    powerline_freqs = (60, 120, 240)
    raw_notch = raw_bandpass.copy().notch_filter(freqs=powerline_freqs)
    report.add_raw(raw=raw_notch, title="Powerline Notch Filter", psd=True)

    ### Identify Bad Channels ###
    nd = NoisyChannels(raw_notch)
    nd.find_bad_by_ransac(channel_wise=True)
    raw_notch.info['bads'] = nd.bad_by_ransac
    report.add_raw(raw=raw_notch, title="Bad Channel Removal (RANSAC)", psd=True)

    ### Average Reference ###
    raw_average_ref = raw_notch.copy().set_eeg_reference('average')
    report.add_raw(raw=raw_average_ref, title="Average Reference", psd=True)

    ### ICA Denoising ###
    ica = mne.preprocessing.ICA(max_iter="auto")

    # Make a copy with 1Hz highpass to remove drift for ICA
    raw_filt = raw_average_ref.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(raw_filt)

    eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw_average_ref, ch_name=["Fp1", 'Fp2'])

    eog_components, eog_scores = ica.find_bads_eog(
    inst=eog_epochs,
    ch_name=["Fp1", 'Fp2'],
    )

    ica.exclude = eog_components

    report.add_ica(
    ica=ica,
    title="ICA Denoising",
    picks=ica.exclude,  # plot the excluded EOG components
    inst=raw_average_ref,
    eog_evoked=eog_epochs.average(),
    eog_scores=eog_scores
    )

    ### Save MNE Report ###
    report.save(f"sub-{subject}_preprocessing_report.html", overwrite=True)

### Start Pre-processing ###

if __name__ == "__main__":
    main(sys.argv[1:])
