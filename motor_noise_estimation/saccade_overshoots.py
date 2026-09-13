#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import config as cfg
import os
from types import SimpleNamespace
from scipy.stats import kruskal, mannwhitneyu
import pandas as pd

# Get the saccade overshoot percentages for zebrafish and macaque saccades
print('Overshoots for zebrafish and macaque saccades')

# Load data and pack it to a SimpleNamespace for easier access
with np.load(os.path.join(cfg.save_path, 'macaque_saccade_data_processed.npz'), allow_pickle=True) as loader_mac:
    mac_data = SimpleNamespace(**loader_mac)
with np.load(os.path.join(cfg.save_path, 'zebrafish_saccade_data_processed.npz'), allow_pickle=True) as loader_zfish:
    zfish_data = SimpleNamespace(**loader_zfish)


# MACAQUE
print('MACAQUE')
# Normalize saccades to be later averaged & get the respective overshoot distribution TODO goes to the overshoot script
# Preallocate saccade arrays for averaging
avgsacmac_dl = np.zeros((mac_data.dl.shape[0], cfg.avgsacdurmac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
avgsacmac_dr = np.zeros((mac_data.dr.shape[0], cfg.avgsacdurmac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
avgsacmac_ul = np.zeros((mac_data.ul.shape[0], cfg.avgsacdurmac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
avgsacmac_ur = np.zeros((mac_data.ur.shape[0], cfg.avgsacdurmac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
# Preallocate saccade arrays for overshoot
overshootsdl = np.zeros(mac_data.dl.shape[0])
overshootsdr = np.zeros(mac_data.dr.shape[0])
overshootsul = np.zeros(mac_data.ul.shape[0])
overshootsur = np.zeros(mac_data.ur.shape[0])

for sac, ons, offs, avgs, overs, direc in zip([mac_data.dl, mac_data.dr, mac_data.ul, mac_data.ur],
                      [mac_data.onsetdl, mac_data.onsetdr, mac_data.onsetul, mac_data.onsetur],
                      [mac_data.offsetdl, mac_data.offsetdr, mac_data.offsetul, mac_data.offsetur],
                      [avgsacmac_dl, avgsacmac_dr, avgsacmac_ul, avgsacmac_ur],
                      [overshootsdl, overshootsdr, overshootsul, overshootsur],
                      ['down left', 'down right', 'up left', 'up right']):
    print(f'Normalizing saccades and getting the overshoot distribution for {direc}')

    for i, tr in enumerate(sac):
        # Normalize tr to have 0 at presaccade and 1 at post-saccade on average
        tr = (tr - tr[ons[i]-cfg.pre_post_dur:ons[i]].mean()) / (tr[offs[i]:cfg.pre_post_dur+offs[i]].mean() - tr[ons[i]-cfg.pre_post_dur:ons[i]].mean())
        # Take 50 ms before and after the saccade
        tr = tr[ons[i]-cfg.pre_post_dur:offs[i]+cfg.pre_post_dur]
        # Update the arrays over which average will be taken
        avgs[i, :len(tr)] = tr
        # Get the overshoot
        overs[i] = np.nanmax(tr)-1

#check if distributions differ:
stats = kruskal(overshootsdl, overshootsdr, overshootsul, overshootsur)
print(f'Kruskal-Wallis test for macaque overshoot distribution: {stats}')

# Do the same analysis for macaque but with a longer time window to be aligned with the zebrafish data (for plotting etc)
# Normalize saccades to be later averaged & get the respective overshoot distribution
# Preallocate saccade arrays for averaging
avgsacmac_dl_long = np.zeros((mac_data.dl.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc
avgsacmac_dr_long = np.zeros((mac_data.dr.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc
avgsacmac_ul_long = np.zeros((mac_data.ul.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc
avgsacmac_ur_long = np.zeros((mac_data.ur.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc

for sac, ons, offs, avgs, direc in zip([mac_data.dl, mac_data.dr, mac_data.ul, mac_data.ur],
                      [mac_data.onsetdl, mac_data.onsetdr, mac_data.onsetul, mac_data.onsetur],
                      [mac_data.offsetdl, mac_data.offsetdr, mac_data.offsetul, mac_data.offsetur],
                      [avgsacmac_dl_long, avgsacmac_dr_long, avgsacmac_ul_long, avgsacmac_ur_long],
                      ['down left', 'down right', 'up left', 'up right']):
    print(f'Normalizing saccades and getting the overshoot distribution for {direc}')

    for i, tr in enumerate(sac):
        # Normalize tr to have 0 at presaccade and 1 at post-saccade on average
        tr = (tr - tr[ons[i]-cfg.pre_post_dur:ons[i]].mean()) / (tr[offs[i]:cfg.pre_post_dur+offs[i]].mean() - tr[ons[i]-cfg.pre_post_dur:ons[i]].mean())
        tr = tr[ons[i]-cfg.pre_post_dur:offs[i]+cfg.pre_post_dur+cfg.extra_t_mac]
        # Update the arrays over which average will be taken
        avgs[i, :len(tr)] = tr



# ZEBRAFISH
print('ZEBRAFISH')
sacextr = np.zeros((len(zfish_data.tracesnt)+len(zfish_data.tracestn), 1001)) * np.nan # 200 ms before 200 ms saccade 200 ms after
for idx, trace in enumerate(zfish_data.tracesnt):
    trace = trace.copy()
    onset, offset = zfish_data.onsetnt[idx], zfish_data.offsetnt[idx]

    if onset < cfg.pre_post_dur:
        # Take whatever you can if the onset is too early
        ons = 0
        addidx = cfg.pre_post_dur - onset
    else:
        ons = onset - cfg.pre_post_dur
        addidx = 0

    if len(trace)-offset < cfg.pre_post_dur:
        # Take whatever you can if the offset is too late
        offs = trace.shape[0]
    else:
        offs = offset + cfg.pre_post_dur
    # Normalize the saccade where average is zero 200 ms before saccade onset and 1 200 ms after saccade offset
    trace -= np.mean(trace[ons:onset])
    trace /= np.mean(trace[offset:offs])
    sac = trace[ons:offs]
    sacextr[idx, addidx:len(sac)+addidx] = sac

for jdx, trace in enumerate(zfish_data.tracestn):
    trace = trace.copy()
    onset, offset = zfish_data.onsettn[jdx], zfish_data.offsettn[jdx]
    # print(onset, offset)
    if onset < cfg.pre_post_dur:
        # Take whatever you can if the onset is too early
        ons = 0
        addidx = cfg.pre_post_dur - onset
    else:
        ons = onset - cfg.pre_post_dur
        addidx = 0

    if len(trace)-offset < cfg.pre_post_dur:
        # Take whatever you can if the offset is too late
        offs = trace.shape[0]
    else:
        offs = offset + cfg.pre_post_dur
    # Normalize the saccade where average is zero 200 ms before saccade onset and 1 200 ms after saccade offset
    trace -= np.mean(trace[ons:onset])
    trace /= np.mean(trace[offset:offs])
    sac = trace[ons:offs]
    sacextr[idx+jdx+1, addidx:len(sac)+addidx] = sac

#Overshoot distribution for different nasal-temporal / temporal-nasal
overshootstn = np.nanmax(sacextr[:len(zfish_data.tracestn)], axis=1) - 1
overshootsnt = np.nanmax(sacextr[len(zfish_data.tracesnt):], axis=1) - 1
overshoots = np.nanmax(sacextr, axis=1) - 1

_, overp = mannwhitneyu(overshootstn, overshootsnt)
print(f'Mann Whitney U test for zebrafish overshoot distribution: {overp}')

# DERIVATIVES
# ZFISH
# NORMALIZE SACCADES
sacextr_deriv = np.zeros((len(zfish_data.tracesnt)+len(zfish_data.tracestn), 1001)) * np.nan # 200 ms before 200 ms saccade 200 ms after
for idx, trace in enumerate(zfish_data.tracesnt):
    trace = trace.copy()
    onset, offset = zfish_data.onsetnt[idx], zfish_data.offsetnt[idx]

    if onset < cfg.pre_post_dur:
        # Take whatever you can if the onset is too early
        ons = 0
        addidx = cfg.pre_post_dur - onset
    else:
        ons = onset - cfg.pre_post_dur
        addidx = 0

    if len(trace)-offset < cfg.pre_post_dur:
        # Take whatever you can if the offset is too late
        offs = trace.shape[0]
    else:
        offs = offset + cfg.pre_post_dur

    sac = trace[ons:offs]
    sacextr_deriv[idx, addidx:len(sac)+addidx] = sac

for jdx, trace in enumerate(zfish_data.tracestn):
    trace = trace.copy()
    onset, offset = zfish_data.onsettn[jdx], zfish_data.offsettn[jdx]
    # print(onset, offset)
    if onset < cfg.pre_post_dur:
        # Take whatever you can if the onset is too early
        ons = 0
        addidx = cfg.pre_post_dur - onset
    else:
        ons = onset - cfg.pre_post_dur
        addidx = 0

    if len(trace)-offset < cfg.pre_post_dur:
        # Take whatever you can if the offset is too late
        offs = trace.shape[0]
    else:
        offs = offset + cfg.pre_post_dur
    sac = trace[ons:offs]
    sacextr_deriv[idx+jdx+1, addidx:len(sac)+addidx] = sac
outlieridx = np.argwhere(sacextr==np.nanmax(sacextr))[0][0]
sacextr_outlierrm_deriv = sacextr_deriv.copy()
sacextr_outlierrm_deriv[outlieridx] = np.nan


# MACAQUE
avgsacmac_dl_long_deriv = np.zeros((mac_data.dl.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
avgsacmac_dr_long_deriv = np.zeros((mac_data.dr.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
avgsacmac_ul_long_deriv = np.zeros((mac_data.ul.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)
avgsacmac_ur_long_deriv = np.zeros((mac_data.ur.shape[0], cfg.avgsacdurmac+cfg.extra_t_mac)) * np.nan # Extracted saccades to get the average etc (very cheap trick xDD)

for sac, ons, offs, avgs, direc in zip([mac_data.dl, mac_data.dr, mac_data.ul, mac_data.ur],
                      [mac_data.onsetdl, mac_data.onsetdr, mac_data.onsetul, mac_data.onsetur],
                      [mac_data.offsetdl, mac_data.offsetdr, mac_data.offsetul, mac_data.offsetur],
                      [avgsacmac_dl_long_deriv, avgsacmac_dr_long_deriv, avgsacmac_ul_long_deriv, avgsacmac_ur_long_deriv],
                      ['down left', 'down right', 'up left', 'up right']):
    for i, tr in enumerate(sac):
        # Take ALL for postsaccade
        tr = tr[ons[i]-cfg.pre_post_dur:offs[i]+cfg.pre_post_dur+cfg.extra_t_mac]
        # Update the arrays over which average will be taken
        avgs[i, :len(tr)] = tr
        # Get the overshoot

deriv_mac = np.gradient(np.concatenate([avgsacmac_dl_long_deriv, avgsacmac_ul_long_deriv, avgsacmac_dr_long_deriv, avgsacmac_ur_long_deriv]),
                        cfg.dt/1000, axis=1) # [°/s]
# deriv_mac = np.nanmean(np.abs(deriv_mac), axis=0)
deriv_mac_l = np.gradient(np.concatenate([avgsacmac_dl_long_deriv, avgsacmac_ul_long_deriv]), cfg.dt/1000, axis=1) # [°/s]
# deriv_mac_l = np.nanmean(np.abs(deriv_mac_l), axis=0)
deriv_mac_r = np.gradient(np.concatenate([avgsacmac_dr_long_deriv, avgsacmac_ur_long_deriv]), cfg.dt/1000, axis=1) # [°/s]
# deriv_mac_r = np.nanmean(np.abs(deriv_mac_r), axis=0)
# zfish
deriv_zfish = np.gradient(sacextr_outlierrm_deriv, cfg.dt/1000, axis=1)  # [°/s]
# deriv_zfish = np.nanmean(np.abs(deriv_zfish), axis=0)  # [°/s]
deriv_zfish_nt = np.gradient(sacextr_outlierrm_deriv[:len(zfish_data.tracesnt)], cfg.dt/1000, axis=1)  # [°/s]
# deriv_zfish_nt = np.nanmean(np.abs(deriv_zfish_nt), axis=0)  # [°/s]
deriv_zfish_tn = np.gradient(sacextr_outlierrm_deriv[len(zfish_data.tracesnt):], cfg.dt/1000, axis=1)  # [°/s]
# deriv_zfish_tn = np.nanmean(np.abs(deriv_zfish_tn), axis=0)  # [°/s]



# SAVE DATA
# ----------------
# Save the raw overshoot data for later analysis
np.savez(os.path.join(cfg.save_path, 'overshoot_data.npz'),
        overshootsdl=overshootsdl, overshootsdr=overshootsdr, overshootsul=overshootsul, overshootsur=overshootsur,
        overshootstn=overshootstn, overshootsnt=overshootsnt, overshoots=overshoots, sacextr=sacextr,
        avgsacmac_dl=avgsacmac_dl, avgsacmac_dr=avgsacmac_dr, avgsacmac_ul=avgsacmac_ul, avgsacmac_ur=avgsacmac_ur,
        avgsacmac_dl_long=avgsacmac_dl_long, avgsacmac_dr_long=avgsacmac_dr_long, avgsacmac_ul_long=avgsacmac_ul_long, avgsacmac_ur_long=avgsacmac_ur_long,
        deriv_mac=deriv_mac, deriv_mac_l=deriv_mac_l, deriv_mac_r=deriv_mac_r,
        deriv_zfish=deriv_zfish, deriv_zfish_nt=deriv_zfish_nt, deriv_zfish_tn=deriv_zfish_tn,
        sacextr_deriv=sacextr_deriv, sacextr_outlierrm_deriv=sacextr_outlierrm_deriv)


# Overshoot distributions
# Create dataframe
ovdfr = pd.DataFrame({'overshoot': np.concatenate([overshootsdl, overshootsdr, overshootsul, overshootsur]),
                      'direction': ['down_left']*len(overshootsdl) + ['down_right']*len(overshootsdr) + ['up_left']*len(overshootsul) + ['up_right']*len(overshootsur),
                      'animal': ['macaque']*len(overshootsdl) + ['macaque']*len(overshootsdr) + ['macaque']*len(overshootsul) + ['macaque']*len(overshootsur)})
# Add zebrafish overshoots
ovdfr = pd.concat([ovdfr,pd.DataFrame({'overshoot': overshoots,
                                    'direction': np.repeat(['nasal_temporal','temporal_nasal'], [len(zfish_data.tracesnt), len(zfish_data.tracestn)]),
                                    'animal': ['zebrafish']*len(overshoots)})], ignore_index=True)
# Save overshoots as csv
ovdfr.to_csv(os.path.join(cfg.save_path, 'overshoots.csv'), index=False)
