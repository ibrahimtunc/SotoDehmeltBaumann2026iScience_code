#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime
import pandas as pd
import config as cfg

# Check the file naming to make sure you can compare between animals.
fdir = cfg.zifsh_path
files = os.listdir(fdir)
dates, fishes = np.zeros(len(files), dtype=object)*np.nan, np.zeros(len(files), dtype=int)*np.nan
for i, f in enumerate(files):
    f = f.strip('.txt')
    f = f.strip('eyepos')
    # print(f)
    components = f.split('_')
    for comp in components:
        if comp.endswith(('20','30')) and len(comp) == 6:
            dates[i] = datetime.strptime(comp, "%m%d%y").strftime("%Y/%m/%d")
        elif comp.startswith('fish'):
            fishes[i] = int(comp.strip('fish'))

# Create a dataframe for the metadata
dfr = pd.DataFrame({'file': files, 'date': dates, 'fish_idx': fishes})

# Print the cases missing a fish index
print(dfr[pd.isnull(dfr['fish_idx'])].to_string(index=False))
# Assume the index is the same for all the fish on the same date & assume 30 is typo for 2020
# Convert to date oject first for simplicity
dfr['date'] = pd.to_datetime(dfr['date'])
# Replace the typo
dfr.loc[dfr['date'].dt.year == 2030, 'date'] = dfr.loc[dfr['date'].dt.year == 2030, 'date'].apply(lambda x: x.replace(year=2020))
# Convert back to str
dfr['date'] = dfr['date'].dt.strftime('%Y/%m/%d')
# Assume all the same fish on this missing day
dfr.loc[pd.isnull(dfr['fish_idx']), 'fish_idx'] = 1
# Get all the index / date combinations
instances = dfr.groupby(['fish_idx', 'date']).size().reset_index(name='count')
instances.fish_idx = instances.fish_idx.astype(int)
print(instances.to_string(index=False))

# Save the data one directory up from the data directory
dfr.to_excel(os.path.join(os.path.dirname(fdir), 'zebrafish_metadata.xlsx'), index=False)

