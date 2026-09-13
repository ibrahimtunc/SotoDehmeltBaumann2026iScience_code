#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


# Configuration file containing the parameters for the motor noise estimation analysis.


# File pathss and related parameters
zifsh_path = 'data/zebrafish_data'  # Path to the directory containing the zebrafish data files
macaque_path = 'data/macaque_data'  # Path to the directory containing the macaque data files
save_path = 'data/processed_data'  # Path to the directory where processed data will be saved
fsd = 'figures' # Figure save directory
exts = ['.png', '.svg', '.pdf'] # List of file extensions for saving figures

for dirs in [save_path, fsd]:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    if dirs == fsd:
        for sd in ('zebrafish', 'macaque', 'joint'):
            if not os.path.exists(os.path.join(dirs, sd)):
                os.makedirs(os.path.join(dirs, sd))

# Saccade detection and noise estimation parameters
pre_post_dur = 50 # Pre and post duration for average saccade in ms
ucut = 0.01 # Cutoff for the command signal magnitude (u) this is the derivative of the saccade in deg/ms
rf = 1000 #sampling rate in Hz.
savgollength = 51 #The vindow length of savgol filter, must be odd number. NOTE IN PAPER LENGTH IS 2 x min saccade duration
savgolorder = 3 # TEST USING THE SAME FOR ZFISH (order 3) which seems to change nothing so far.
RMSfac = -9999
dt = 1 # Time step in ms, this is the inverse of the sampling rate (1/fs) in ms

# Animal-specific parameters
# MACAQUE parameters
# Dictionary for macaque saccade on - offset detection parameters
macdet = {'savgollength' : savgollength,
          'savgolorder' : 3,
          'a' : 0.8,
          'b' : 0.5,
          'onstd' : 10,
          'macaque' : True}
avgsacdurmac = 150 # Average saccade duration to be considered in ms
extra_t_mac = 80 # Required for plotting etc.


# ZEBRAFISH parameters
angthres = 5
flthres = 0.3
saconsmoothsigma = 15 #gaussian smoothing filter standard deviation for saccade onset detection
# savgolorder = 4 #order of the polynomial fit used to derive the savgol filter. DIFFERENT THAN MACAQUE! (3 vs 4) NOTE IN PAPER ORDER IS 2
nperbin = 100  # number of datapoints per bin for the multiplkicative noise estimation, this is for the u.
onsrem = 10  # number of datapoints to remove from saccade start
offsrem = 150  # number of datapoints to remove from saccade end

