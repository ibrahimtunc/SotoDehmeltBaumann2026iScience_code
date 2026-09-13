#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import runpy

# RUN the entire motor noise estimation analysis pipeline for zebrafish and macaque data.
# PREPROCESS
runpy.run_path('preprocess_saccade_data.py')
# ESTIMATE NOISE
runpy.run_path('noise_estimation.py')
# OVERSHOOT DISTRIBUTION
runpy.run_path('saccade_overshoots.py')
# PLOT GENERATION
runpy.run_path('plot_figures.py')