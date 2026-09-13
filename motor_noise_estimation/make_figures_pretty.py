#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# Adjust the default parameters dict for matplotlib to make figures look better. You can simply import this script and it will automatically update the default parameters.
figdict = {'axes.titlesize': 25,
           'axes.labelsize': 25,
           'xtick.labelsize': 25,
           'ytick.labelsize': 25,
           'legend.fontsize': 25,
           'figure.titlesize': 30,
           'image.cmap': 'gray',
           'axes.formatter.limits': [-7, 7],
           'font.size': 25,
           'axes.spines.top': False,
           'axes.spines.right': False,
           'axes.linewidth': 2,
           'ytick.major.size': 4,
           'xtick.major.size': 4,
           'ytick.major.width': 1.5,
           'xtick.major.width': 1.5,
           'legend.handlelength': 1.5,
           'legend.columnspacing': 0.75,
           'legend.handletextpad': 0.4,
           'legend.frameon': False,
           'hatch.linewidth': 2}

plt.rcParams.update(figdict)
