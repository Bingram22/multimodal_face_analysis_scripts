'''

Stage 1: Pre-processing

This script performs pre-processing on the EEG data. The pipeline
consists of bandpass filtering, notch filtering to remove line noise,
bad channel rejection with RANSAC, and ICA denoising. Cleaned
raw and epochs are saved in the MNE bids derivative folder.

'''

### Import Packages ###
import mne
import re
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
    get_entity_vals(bids_root, "session")
    datatype = "meg"
    extensions = [".fif", ".tsv"]  # ignore .json files
    bids_path = BIDSPath(root=bids_root, datatype=datatype)
    task = "facerecognition"
    suffix = "meg"
    bids_path = bids_path.update(subject=subject, task=task, suffix=suffix, session = 'meg', run = '01')

    # Read Raw
    raw = read_raw_bids(bids_path=bids_path, verbose=False, extra_params = {'preload' : True})
    raw.resample(128)
    raw.set_channel_types({'EEG061' : 'eog', 'EEG062' : 'eog', 'EEG063' : 'ecg', 'EEG064' : 'misc'})

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

    ecg_components, ecg_scores = ica.find_bads_ecg(raw_average_ref)
    eog_components, eog_scores = ica.find_bads_eog(raw_average_ref)

    ica.exclude = eog_components + ecg_components

    report.add_ica(
    ica=ica,
    title="ICA Denoising",
    picks=ica.exclude,  # plot the excluded EOG components
    inst=raw_average_ref,
    eog_scores=eog_scores,
    ecg_scores=ecg_scores
    )

    # Remove bad components
    ica.apply(raw_average_ref)

    ### Save MNE Report ###

    # Define the output directory
    output_dir = f"outputs/01_preprocessing/sub-{subject}/"

    report_path = os.path.join(output_dir, f"sub-{subject}_preprocessing_report.html")
    fif_path = os.path.join(output_dir, f"sub-{subject}_preprocessed_data.fif")

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the MNE Report
    report.save(report_path, overwrite=True)

    # Save the MNE object (e.g., Raw, Epochs)
    raw_average_ref.save(fif_path, overwrite=True)

### Start Pre-processing ###

if __name__ == "__main__":
    main(re.search(r'\d+', sys.argv[1]).group())
