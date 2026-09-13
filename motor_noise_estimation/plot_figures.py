#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import make_figures_pretty
import os
import config as cfg
from types import SimpleNamespace
import numpy as np
import function_scripts as fnc

# Plot figures based on the analysis results
print('Plotting figures based on the analysis results')

# Load all data from the analysis results
with np.load(os.path.join(cfg.save_path, 'macaque_saccade_data_processed.npz'), allow_pickle=True) as loader_mac:
    mac_data = SimpleNamespace(**loader_mac)
with np.load(os.path.join(cfg.save_path, 'zebrafish_saccade_data_processed.npz'), allow_pickle=True) as loader_zfish:
    zfish_data = SimpleNamespace(**loader_zfish)
with np.load(os.path.join(cfg.save_path, 'overshoot_data.npz'), allow_pickle=True) as loader_ovr:
    overshoot_data = SimpleNamespace(**loader_ovr)
# Load noise estimates for both macaque and zebrafish
with np.load(os.path.join(cfg.save_path, 'macaque_noise_estimates.npz'), allow_pickle=True) as loader_mne:
    mac_data = SimpleNamespace(**loader_mne, **mac_data.__dict__)
with np.load(os.path.join(cfg.save_path, 'zebrafish_noise_estimates.npz'), allow_pickle=True) as loader_zne:
    zfish_data = SimpleNamespace(**loader_zne, **zfish_data.__dict__)


# PLOTTING
# ----------------
# Figure (1) SACCADE EXAMPLES -> Plot example saccades sorted by the fit quality i.e. RMS minimum, 3-4 examples for zfish and macaque. DONE
example_num = 4 # Number of examples to plot
# 1) MACAQUE -> 3 examples each for left and right directions
# a) Left
RMS_left = np.concatenate([mac_data.restdl.item()['RMS'], mac_data.restul.item()['RMS']])
fitsparams_left = np.concatenate([mac_data.restdl.item()['fitparams'], mac_data.restul.item()['fitparams']])
traces_left = np.concatenate([mac_data.dl, mac_data.ul], axis=0)
ons_left = np.concatenate([mac_data.onsetdl, mac_data.onsetul])
offs_left = np.concatenate([mac_data.offsetdl, mac_data.offsetul])
# b) Right
RMS_right = np.concatenate([mac_data.restdr.item()['RMS'], mac_data.restur.item()['RMS']])
fitsparams_right = np.concatenate([mac_data.restdr.item()['fitparams'], mac_data.restur.item()['fitparams']])
traces_right = np.concatenate([mac_data.dr, mac_data.ur], axis=0)
ons_right = np.concatenate([mac_data.onsetdr, mac_data.onsetur])
offs_right = np.concatenate([mac_data.offsetdr, mac_data.offsetur])
# Choose top 3 lowest RMS
RMS_left_idx = np.argsort(RMS_left)[:example_num]
RMS_right_idx = np.argsort(RMS_right)[:example_num]

for i in range(RMS_left_idx.shape[0]):
    # LEFT
    idx_l = RMS_left_idx[i]
    figsac, axsac = plt.subplots(figsize=(8.7, 6.6))
    t_sac_tot = np.arange(traces_left[idx_l, ons_left[idx_l]-cfg.pre_post_dur:offs_left[idx_l]+cfg.pre_post_dur].shape[0])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, traces_left[idx_l, ons_left[idx_l]-cfg.pre_post_dur:offs_left[idx_l]+cfg.pre_post_dur], 'k-', label='Saccade')
    fit = fnc.saccade_fit_func(t_sac_tot, *fitsparams_left[idx_l])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, fit, 'r--', label='Fit')

    # Adjust figure
    axsac.set_xlabel('t [ms]')
    axsac.set_ylabel('Eye position [°]')
    axsac.set_title('Left')
    axsac.text(0.75, 0.5, f'RMS: {RMS_left[idx_l]:.5f}', ha='center', va='center', transform=axsac.transAxes)
    figsac.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
    axsac.legend()
    plt.pause(0.001)
    # SAVE FIGURE AND CLOSE
    [figsac.savefig(os.path.join(cfg.fsd, 'macaque/saccade_example_left_%d%s' % (i, ext))) for ext in cfg.exts]
    plt.close(figsac)

    # RIGHT
    idx_r = RMS_right_idx[i]
    figsac, axsac = plt.subplots(figsize=(8.7, 6.6))
    t_sac_tot = np.arange(traces_right[idx_r, ons_right[idx_r]-cfg.pre_post_dur:offs_right[idx_r]+cfg.pre_post_dur].shape[0])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, traces_right[idx_r, ons_right[idx_r]-cfg.pre_post_dur:offs_right[idx_r]+cfg.pre_post_dur], 'k-', label='Saccade')
    fit = fnc.saccade_fit_func(t_sac_tot, *fitsparams_right[idx_r])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, fit, 'r--', label='Fit')
    # Adjust figure
    axsac.set_xlabel('t [ms]')
    axsac.set_ylabel('Eye position [°]')
    axsac.set_title('Right')
    axsac.text(0.75, 0.5, f'RMS: {RMS_right[idx_r]:.5f}', ha='center', va='center', transform=axsac.transAxes)
    figsac.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
    axsac.legend()
    plt.pause(0.001)
    # SAVE FIGURE AND CLOSE
    [figsac.savefig(os.path.join(cfg.fsd, 'macaque/saccade_example_right_%d%s' % (i, ext))) for ext in cfg.exts]
    plt.close(figsac)

# 2) ZEBRAFISH -> 3 examples each for TN and NT directions
RMS_tn_idx = np.argsort(zfish_data.RMSstn)[:example_num]
RMS_nt_idx = np.argsort(zfish_data.RMSsnt)[:example_num]

for i in range(RMS_tn_idx.shape[0]):
    # TN
    idx_tn = RMS_tn_idx[i]
    figsac, axsac = plt.subplots(figsize=(8.7, 6.6))
    t_sac_tot = np.arange(zfish_data.tracestn[idx_tn][zfish_data.onsettn[idx_tn]-cfg.pre_post_dur:zfish_data.offsettn[idx_tn]+cfg.pre_post_dur].shape[0])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, zfish_data.tracestn[idx_tn][zfish_data.onsettn[idx_tn]-cfg.pre_post_dur:zfish_data.offsettn[idx_tn]+cfg.pre_post_dur],
               'k-', label='Saccade')
    fit = fnc.saccade_fit_func(t_sac_tot, *zfish_data.fitparamstn[idx_tn])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, fit, 'r--', label='Fit')

    # Adjust figure
    axsac.set_xlabel('t [ms]')
    axsac.set_ylabel('Eye position [°]')
    axsac.set_title('Temporal to Nasal')
    axsac.text(0.75, 0.5, f'RMS: {zfish_data.RMSstn[idx_tn]:.5f}', ha='center', va='center', transform=axsac.transAxes)
    figsac.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
    axsac.legend()
    axsac.set_yticks(np.round(axsac.get_yticks()))
    plt.pause(0.001)
    # SAVE FIGURE AND CLOSE
    [figsac.savefig(os.path.join(cfg.fsd, 'zebrafish/saccade_example_temporal_to_nasal_%d%s' % (i, ext))) for ext in cfg.exts]
    plt.close(figsac)

    # NT
    idx_nt = RMS_nt_idx[i]
    figsac, axsac = plt.subplots(figsize=(8.7, 6.6))
    t_sac_tot = np.arange(zfish_data.tracesnt[idx_nt][zfish_data.onsetnt[idx_nt]-cfg.pre_post_dur:zfish_data.offsetnt[idx_nt]+cfg.pre_post_dur].shape[0])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, zfish_data.tracesnt[idx_nt][zfish_data.onsetnt[idx_nt]-cfg.pre_post_dur:zfish_data.offsetnt[idx_nt]+cfg.pre_post_dur],
               'k-', label='Saccade')
    fit = fnc.saccade_fit_func(t_sac_tot, *zfish_data.fitparamsnt[idx_nt])
    axsac.plot(t_sac_tot-cfg.pre_post_dur, fit, 'r--', label='Fit')
    # Adjust figure
    axsac.set_xlabel('t [ms]')
    axsac.set_ylabel('Eye position [°]')
    axsac.set_title('Nasal to Temporal')
    axsac.text(0.75, 0.5, f'RMS: {zfish_data.RMSsnt[idx_nt]:.5f}', ha='center', va='center', transform=axsac.transAxes)
    figsac.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
    axsac.legend()
    axsac.set_yticks(np.round(axsac.get_yticks()))
    plt.pause(0.001)
    # SAVE FIGURE AND CLOSE
    [figsac.savefig(os.path.join(cfg.fsd, 'zebrafish/saccade_example_nasal_to_temporal_%d%s' % (i, ext))) for ext in cfg.exts]
    plt.close(figsac)

# Figure 2 All saccades & average and shade DONE
# 1) MACAQUE
macaque_t = np.arange(-cfg.pre_post_dur, cfg.avgsacdurmac - cfg.pre_post_dur)
# a) Not merged
figsaclr, axsaclr = plt.subplots(1, 2, figsize=(10.91, 6.6), sharex=True, sharey=True)
axsaclr[0].plot(macaque_t, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0), 'r-', label='Mean', zorder=3)
axsaclr[0].fill_between(macaque_t,
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0) + np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0),
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0) - np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0),
    color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsaclr[0].plot(macaque_t, np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0).T, '-', alpha=0.25, zorder=1, color='gray')
axsaclr[1].plot(macaque_t, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0), 'r-', label='Mean', zorder=3)
axsaclr[1].fill_between(macaque_t,
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) + np.nanstd(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) - np.nanstd(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsaclr[1].plot(macaque_t, np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0).T, '-', alpha=0.25, zorder=1, color='gray')
axsaclr[1].plot([], '-', alpha=0.5, label='Trial', color='gray')

# Adjust figure
# Axis labels
axsaclr[0].set_xlabel('t [ms]', x=1)
axsaclr[0].set_ylabel('Eye position (norm)')
# Title
for i, tit in enumerate(['Left', 'Right']):
    axsaclr[i].set_title(tit)
# Legend etc.
axsaclr[1].legend(loc=[0.6,0.3])
figsaclr.subplots_adjust(top=0.9, bottom=0.13, left=0.105, right=0.97, hspace=0.18, wspace=0.12)
# Save figure
plt.pause(0.001)
# Save figure
[figsaclr.savefig(os.path.join(cfg.fsd, 'macaque/all_saccades_lr_separated%s' % ext)) for ext in cfg.exts]
plt.close(figsaclr)

# b) Merged
figsacmer, axsacmer = plt.subplots(1,1, figsize=(8.7, 6.6))
axsacmer.plot(macaque_t, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
              'r-', label='Mean', zorder=3)
axsacmer.fill_between(macaque_t,
                        np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) +
                        np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
                        np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) -
                        np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsacmer.plot(macaque_t, np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0).T,
              '-', alpha=0.25, zorder=1, color='gray')
axsacmer.plot([], '-', alpha=0.5, label='Trial', color='gray')

# Adjust figure
# Axis labels
axsacmer.set_xlabel('t [ms]')
axsacmer.set_ylabel('Eye position (norm)')
# Legend etc.
axsacmer.legend(loc=[0.6,0.3])
figsacmer.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
# Save figure
plt.pause(0.001)
[figsacmer.savefig(os.path.join(cfg.fsd, 'macaque/all_saccades_merged%s' % ext)) for ext in cfg.exts]
plt.close(figsacmer)

# 2) ZEBRAFISH
zfish_t = np.arange(-cfg.pre_post_dur, overshoot_data.sacextr.shape[1] - cfg.pre_post_dur)
outlieridx = np.argwhere(overshoot_data.sacextr==np.nanmax(overshoot_data.sacextr))[0][0]
sacextr_outlierrm = overshoot_data.sacextr.copy()
sacextr_outlierrm[outlieridx] = np.nan
# a) Not merged
figsaclr, axsaclr = plt.subplots(1, 2, figsize=(10.91, 6.6), sharex=True, sharey=True)
axsaclr[0].plot(zfish_t, np.nanmean(overshoot_data.sacextr[:len(zfish_data.tracesnt)], axis=0), 'r-', label='Mean', zorder=3)
axsaclr[0].fill_between(zfish_t,
                        np.nanmean(overshoot_data.sacextr[:len(zfish_data.tracesnt)], axis=0) + np.nanstd(overshoot_data.sacextr[:len(zfish_data.tracesnt)], axis=0),
                        np.nanmean(overshoot_data.sacextr[:len(zfish_data.tracesnt)], axis=0) - np.nanstd(overshoot_data.sacextr[:len(zfish_data.tracesnt)], axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsaclr[0].plot(zfish_t, overshoot_data.sacextr[:len(zfish_data.tracesnt)].T, '-', alpha=0.25, zorder=1, color='gray')
axsaclr[1].plot(zfish_t, np.nanmean(overshoot_data.sacextr[len(zfish_data.tracesnt):], axis=0), 'r-', label='Mean', zorder=3)
axsaclr[1].fill_between(zfish_t,
                        np.nanmean(overshoot_data.sacextr[len(zfish_data.tracesnt):], axis=0) + np.nanstd(overshoot_data.sacextr[len(zfish_data.tracesnt):], axis=0),
                        np.nanmean(overshoot_data.sacextr[len(zfish_data.tracesnt):], axis=0) - np.nanstd(overshoot_data.sacextr[len(zfish_data.tracesnt):], axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsaclr[1].plot(zfish_t, overshoot_data.sacextr[len(zfish_data.tracesnt):].T, '-', alpha=0.25, zorder=1, color='gray')
axsaclr[1].plot([], '-', alpha=0.5, label='Trial', color='gray')
# Adjust figure
# Axis labels
axsaclr[0].set_xlabel('t [ms]', x=1)
axsaclr[0].set_ylabel('Eye position (norm)')
# Title
for i, tit in enumerate(['N - T', 'T - N']):
    axsaclr[i].set_title(tit)
# Legend etc.
axsaclr[1].legend(loc=[0.6,0.3])
figsaclr.subplots_adjust(top=0.9, bottom=0.13, left=0.105, right=0.97, hspace=0.18, wspace=0.12)
# Save figure
plt.pause(0.001)

[figsaclr.savefig(os.path.join(cfg.fsd, 'zebrafish/all_saccades_nt_separated%s' % ext)) for ext in cfg.exts]
plt.close(figsaclr)

# a1) Outlier removed
figsaclr, axsaclr = plt.subplots(1, 2, figsize=(10.91, 6.6), sharex=True, sharey=True)
axsaclr[0].plot(zfish_t, np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0), 'r-', label='Mean', zorder=3)
axsaclr[0].fill_between(zfish_t,
                        np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0) + np.nanstd(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0),
                        np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0) - np.nanstd(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsaclr[0].plot(zfish_t, sacextr_outlierrm[:len(zfish_data.tracesnt)].T, '-', alpha=0.25, zorder=1, color='gray')
axsaclr[1].plot(zfish_t, np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0), 'r-', label='Mean', zorder=3)
axsaclr[1].fill_between(zfish_t,
                        np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0) + np.nanstd(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0),
                        np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0) - np.nanstd(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsaclr[1].plot(zfish_t, sacextr_outlierrm[len(zfish_data.tracesnt):].T, '-', alpha=0.25, zorder=1, color='gray')
axsaclr[1].plot([], '-', alpha=0.5, label='Trial', color='gray')
# Adjust figure
# Axis labels
axsaclr[0].set_xlabel('t [ms]', x=1)
axsaclr[0].set_ylabel('Eye position (norm)')
# Title
for i, tit in enumerate(['N - T', 'T - N']):
    axsaclr[i].set_title(tit)
# Legend etc.
axsaclr[1].legend(loc=[0.6,0.3])
figsaclr.subplots_adjust(top=0.9, bottom=0.13, left=0.105, right=0.97, hspace=0.18, wspace=0.12)
# Save figure
plt.pause(0.001)
[figsaclr.savefig(os.path.join(cfg.fsd, 'zebrafish/all_saccades_nt_separated_outlier_removed%s' % ext)) for ext in cfg.exts]
plt.close(figsaclr)

# b) Merged
figsacmer, axsacmer = plt.subplots(1,1, figsize=(8.7, 6.6))
axsacmer.plot(zfish_t, np.nanmean(overshoot_data.sacextr, axis=0), 'r-', label='Mean', zorder=3)
axsacmer.fill_between(zfish_t,
    np.nanmean(overshoot_data.sacextr, axis=0) + np.nanstd(overshoot_data.sacextr, axis=0), np.nanmean(overshoot_data.sacextr, axis=0) - np.nanstd(overshoot_data.sacextr, axis=0),
    color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsacmer.plot(zfish_t, overshoot_data.sacextr.T, '-', alpha=0.25, zorder=1, color='gray')
axsacmer.plot([], '-', alpha=0.5, label='Trial', color='gray')

# Adjust figure
# Axis labels
axsacmer.set_xlabel('t [ms]')
axsacmer.set_ylabel('Eye position (norm)')
# Legend etc.
axsacmer.legend(loc=[0.6,0.3])
figsacmer.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
# Save figure
plt.pause(0.001)

[figsacmer.savefig(os.path.join(cfg.fsd, 'zebrafish/all_saccades_merged%s' % ext)) for ext in cfg.exts]
plt.close(figsacmer)

# b1) Outlier removed
figsacmer, axsacmer = plt.subplots(1,1, figsize=(8.7, 6.6))
axsacmer.plot(zfish_t, np.nanmean(sacextr_outlierrm, axis=0), 'r-', label='Mean', zorder=3)
axsacmer.fill_between(zfish_t,
                      np.nanmean(sacextr_outlierrm, axis=0) + np.nanstd(sacextr_outlierrm, axis=0),
                      np.nanmean(sacextr_outlierrm, axis=0) - np.nanstd(sacextr_outlierrm, axis=0),
                      color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsacmer.plot(zfish_t, sacextr_outlierrm.T, '-', alpha=0.25, zorder=1, color='gray')
axsacmer.plot([], '-', alpha=0.5, label='Trial', color='gray')
# Adjust figure
# Axis labels
axsacmer.set_xlabel('t [ms]')
axsacmer.set_ylabel('Eye position (norm)')
# Legend etc.
axsacmer.legend(loc=[0.6,0.3])
figsacmer.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
# Save figure
plt.pause(0.001)
[figsacmer.savefig(os.path.join(cfg.fsd, 'zebrafish/all_saccades_merged_outlier_removed%s' % ext)) for ext in cfg.exts]
plt.close(figsacmer)

# Figure 3 Saccade average and shade
# 1) MACAQUE
# a) Not merged
figsacavg, axsacavg = plt.subplots(1, 2, figsize=(10.91, 6.6), sharex=True, sharey=True)
axsacavg[0].plot(macaque_t, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0), 'r-', label='Mean', zorder=3)
axsacavg[0].fill_between(macaque_t,
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0) + np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0),
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0) - np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul], axis=0), axis=0),
    color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsacavg[1].plot(macaque_t, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0), 'r-', label='Mean', zorder=3)
axsacavg[1].fill_between(macaque_t,
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) + np.nanstd(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
    np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) - np.nanstd(np.concatenate([overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
    color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
# Adjust figure
# Axis labels
axsacavg[0].set_xlabel('t [ms]', x=1)
axsacavg[0].set_ylabel('Eye position (norm)')
# Title
for i, tit in enumerate(['Left', 'Right']):
    axsacavg[i].set_title(tit)
# Legend etc.
axsacavg[1].legend(loc=[0.6,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.9, bottom=0.13, left=0.105, right=0.97, hspace=0.18, wspace=0.12)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'macaque/saccade_average_separated%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# b) Merged
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0), 'r-', label='Mean', zorder=3)
axsacavg.fill_between(macaque_t,
                      np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) +
                      np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
                      np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0) -
                      np.nanstd(np.concatenate([overshoot_data.avgsacmac_dl, overshoot_data.avgsacmac_ul, overshoot_data.avgsacmac_dr, overshoot_data.avgsacmac_ur], axis=0), axis=0),
                      color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
# Adjust figure
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Eye position (norm)')
# Legend etc.
axsacavg.legend(loc=[0.6,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'macaque/saccade_average_merged%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)
# 2) ZEBRAFISH
# a) Not merged
figsacavg, axsacavg = plt.subplots(1, 2, figsize=(10.91, 6.6), sharex=True, sharey=True)
axsacavg[0].plot(zfish_t, np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0), 'r-', label='Mean', zorder=3)
axsacavg[0].fill_between(zfish_t,
                        np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0) + np.nanstd(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0),
                        np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0) - np.nanstd(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
axsacavg[1].plot(zfish_t, np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0), 'r-', label='Mean', zorder=3)
axsacavg[1].fill_between(zfish_t,
                        np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0) + np.nanstd(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0),
                        np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0) - np.nanstd(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0),
                        color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
# Adjust figure
# Axis labels
axsacavg[0].set_xlabel('t [ms]', x=1)
axsacavg[0].set_ylabel('Eye position (norm)')
# Title
for i, tit in enumerate(['N - T', 'T - N']):
    axsacavg[i].set_title(tit)
# Legend etc.
axsacavg[1].legend(loc=[0.6,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.9, bottom=0.13, left=0.105, right=0.97, hspace=0.18, wspace=0.12)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'zebrafish/saccade_average_separated%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# b) Merged
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(zfish_t, np.nanmean(sacextr_outlierrm, axis=0), 'r-', label='Mean', zorder=3)
axsacavg.fill_between(zfish_t,
                      np.nanmean(sacextr_outlierrm, axis=0) + np.nanstd(sacextr_outlierrm, axis=0), np.nanmean(sacextr_outlierrm, axis=0) - np.nanstd(sacextr_outlierrm, axis=0),
                      color='k', alpha=0.75, label=r'$\pm$ std', zorder=2)
# Adjust figure
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Eye position (norm)')
# Legend etc.
axsacavg.legend(loc=[0.6,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'zebrafish/saccade_average_merged%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# Figure 4 Saccade overshoot distribution
# 1) MACAQUE
figos, axos = plt.subplots(1,1, figsize=(6, 6))
figos.subplots_adjust(top=0.943, bottom=0.152, left=0.203, right=0.968, hspace=0.2, wspace=0.2)
h = axos.hist(np.concatenate([overshoot_data.overshootsdl, overshoot_data.overshootsdr, overshoot_data.overshootsul, overshoot_data.overshootsur]),
              bins=20, color='k', density=True)
axos.set_ylabel('Density')
axos.set_xlim(-0.0025, 0.045)
axos.set_xlabel('Overshoot [% / 100]')
plt.pause(0.001)
[figos.savefig(os.path.join(cfg.fsd, 'macaque/overshoot_distribution%s' % ext)) for ext in cfg.exts]
plt.close(figos)

# 2) ZEBRAFISH
figos, axos = plt.subplots(1,1, figsize=(6, 6))
figos.subplots_adjust(top=0.943, bottom=0.152, left=0.158, right=0.968, hspace=0.2, wspace=0.2)
h = axos.hist(overshoot_data.overshoots[overshoot_data.overshoots<2], bins=20, color='k', density=True)
axos.set_ylabel('Density')
axos.set_xlabel('Overshoot [% / 100]')
plt.pause(0.001)
[figos.savefig(os.path.join(cfg.fsd, 'zebrafish/overshoot_distribution%s' % ext)) for ext in cfg.exts]
plt.close(figos)

# Figure 5 Average saccades zfish macaque overlaid
# MERGED BOTH
macaque_t_long = np.arange(-cfg.pre_post_dur, cfg.avgsacdurmac + cfg.extra_t_mac - cfg.pre_post_dur)
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t_long,
  np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl_long, overshoot_data.avgsacmac_ul_long, overshoot_data.avgsacmac_dr_long, overshoot_data.avgsacmac_ur_long], axis=0), axis=0),
  'r-', label='Macaque', zorder=3)
axsacavg.plot(zfish_t[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac], np.nanmean(sacextr_outlierrm, axis=0)[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac],
              'b-', label='Zebrafish', zorder=3)
# Adjust figure
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Eye position (norm)')
# Legend etc.
axsacavg.legend(loc=[0.6,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'joint/average_saccades_merged%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# 2) Zfish separated macaque merged
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t_long,
  np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl_long, overshoot_data.avgsacmac_ul_long, overshoot_data.avgsacmac_dr_long, overshoot_data.avgsacmac_ur_long], axis=0), axis=0),
  'r-', label='Macaque', zorder=3)
axsacavg.plot(zfish_t[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac], np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0)[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac],
              'b-', label='Zebrafish N - T', zorder=3)
axsacavg.plot(zfish_t[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac], np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0)[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac],
              'b--', label='Zebrafish T - N', zorder=3)
# Adjust figure
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Eye position (norm)')
# Legend etc.
axsacavg.legend(loc=[0.5,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'joint/average_saccades_separated%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# 3) Zfish separated macaque separated
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t_long, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dl_long, overshoot_data.avgsacmac_ul_long], axis=0), axis=0), 'r-', label='Macaque Left', zorder=3)
axsacavg.plot(macaque_t_long, np.nanmean(np.concatenate([overshoot_data.avgsacmac_dr_long, overshoot_data.avgsacmac_ur_long], axis=0), axis=0), 'r--', label='Macaque Right', zorder=3)
axsacavg.plot(zfish_t[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac], np.nanmean(sacextr_outlierrm[:len(zfish_data.tracesnt)], axis=0)[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac],
              'b-', label='Zebrafish N - T', zorder=3)
axsacavg.plot(zfish_t[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac], np.nanmean(sacextr_outlierrm[len(zfish_data.tracesnt):], axis=0)[zfish_t<cfg.avgsacdurmac+cfg.extra_t_mac],
              'b--', label='Zebrafish T - N', zorder=3)
# Adjust figure
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Eye position (norm)')
# Legend etc.
axsacavg.legend(loc=[0.5,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'joint/average_saccades_separated_both%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# Figure 5.5 BONUS Figure 5 but with absolute velocity in y axis (instead of normalized position, but is velocity normalized?)
# MACAQUE


# 1) MERGED BOTH
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t_long, np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0), 'r-', label='Macaque', zorder=3)
axsacavg.fill_between(macaque_t_long,
                     np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0) + np.nanstd(np.abs(overshoot_data.deriv_mac), axis=0),
                     np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0) - np.nanstd(np.abs(overshoot_data.deriv_mac), axis=0),
                     color='r', zorder=2, alpha=0.25)
axsacavg.plot(zfish_t[zfish_t<300], np.nanmean(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300], 'b-', label='Zebrafish', zorder=3)
axsacavg.fill_between(zfish_t[zfish_t<300],
                        np.nanmean(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300] + np.nanstd(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300],
                        np.nanmean(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300] - np.nanstd(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300],
                        color='b', zorder=2, alpha=0.25)
# Adjust figure
axsacavg.set_ylim(-0.001, axsacavg.get_ylim()[-1])
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Absolute velocity [°/s]')
# Legend etc.
axsacavg.legend(loc=[0.6,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'joint/average_saccades_merged_velocity%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# 2) Zfish separated macaque merged
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t_long, np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0), 'r-', label='Macaque', zorder=3)
axsacavg.plot(zfish_t[zfish_t<300], np.nanmean(np.abs(overshoot_data.deriv_zfish_nt), axis=0)[zfish_t<300], 'b-', label='Zebrafish N - T', zorder=3)
axsacavg.plot(zfish_t[zfish_t<300], np.nanmean(np.abs(overshoot_data.deriv_zfish_tn), axis=0)[zfish_t<300], 'b--', label='Zebrafish T - N', zorder=3)
# Adjust figure
axsacavg.set_ylim(-0.001, axsacavg.get_ylim()[-1])
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Absolute velocity [°/s]')
# Legend etc.
axsacavg.legend(loc=[0.5,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'joint/average_saccades_separated_velocity%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# 3) Zfish separated macaque separated
figsacavg, axsacavg = plt.subplots(1, 1, figsize=(8.7, 6.6))
axsacavg.plot(macaque_t_long, np.nanmean(np.abs(overshoot_data.deriv_mac_l), axis=0), 'r-', label='Macaque Left', zorder=3)
axsacavg.plot(macaque_t_long, np.nanmean(np.abs(overshoot_data.deriv_mac_r), axis=0), 'r--', label='Macaque Right', zorder=3)
axsacavg.plot(zfish_t[zfish_t<300], np.nanmean(np.abs(overshoot_data.deriv_zfish_nt), axis=0)[zfish_t<300], 'b-', label='Zebrafish N - T', zorder=3)
axsacavg.plot(zfish_t[zfish_t<300], np.nanmean(np.abs(overshoot_data.deriv_zfish_tn), axis=0)[zfish_t<300], 'b--', label='Zebrafish T - N', zorder=3)
# Adjust figure
axsacavg.set_ylim(-0.001, axsacavg.get_ylim()[-1])
# Axis labels
axsacavg.set_xlabel('t [ms]')
axsacavg.set_ylabel('Absolute velocity [°/s]')
# Legend etc.
axsacavg.legend(loc=[0.5,0.3])
# Save figure
figsacavg.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacavg.savefig(os.path.join(cfg.fsd, 'joint/average_saccades_separated_both_velocity%s' % ext)) for ext in cfg.exts]
plt.close(figsacavg)

# 1.5) DO FOR BOTH MACAQUE AND ZFISH SEPARATELY THE VELOCITY WITH INDIVIDUAL TRACES AND THE STD ENVELOPE,
# SIMILAR TO FIG 2 i.e. for each spearately else it's going to be very crowded

# MACAQUE
figsacmer, axsacmer = plt.subplots(1,1, figsize=(8.7, 6.6))
axsacmer.plot(macaque_t_long, np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0), 'r-', label='Mean', zorder=3)
axsacmer.fill_between(macaque_t_long,
                     np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0) + np.nanstd(np.abs(overshoot_data.deriv_mac), axis=0),
                     np.nanmean(np.abs(overshoot_data.deriv_mac), axis=0) - np.nanstd(np.abs(overshoot_data.deriv_mac), axis=0),
                     color='k', zorder=2, alpha=0.75, label=r'$\pm$ std')
axsacmer.plot(macaque_t_long, np.abs(overshoot_data.deriv_mac).T, '-', alpha=0.25, zorder=1, color='gray')
axsacmer.plot([], '-', alpha=0.5, label='Trial', color='gray')
# Adjust figure
axsacmer.set_ylim(-0.001, axsacmer.get_ylim()[-1])
# Axis labels
axsacmer.set_xlabel('t [ms]')
axsacmer.set_ylabel('Absolute velocity [°/s]')
# Legend etc.
axsacmer.legend(loc='upper right')
# Save figure
figsacmer.subplots_adjust(top=0.93, bottom=0.135, left=0.155, right=0.95, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacmer.savefig(os.path.join(cfg.fsd, 'macaque/saccade_velocities%s' % ext)) for ext in cfg.exts]
plt.close(figsacmer)

# ZEBRAFISH
figsacmer, axsacmer = plt.subplots(1,1, figsize=(8.7, 6.6))
axsacmer.plot(zfish_t[zfish_t<300], np.nanmean(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300], 'b-', label='Mean', zorder=3)
axsacmer.fill_between(zfish_t[zfish_t<300],
                     np.nanmean(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300] + np.nanstd(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300],
                     np.nanmean(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300] - np.nanstd(np.abs(overshoot_data.deriv_zfish), axis=0)[zfish_t<300],
                     color='k', zorder=2, alpha=0.75, label=r'$\pm$ std')
axsacmer.plot(zfish_t[zfish_t<300], np.abs(overshoot_data.deriv_zfish[:, zfish_t<300]).T, '-', alpha=0.25, zorder=1, color='gray')
axsacmer.plot([], '-', alpha=0.5, label='Trial', color='gray')
# Adjust figure
axsacmer.set_ylim(-0.001, axsacmer.get_ylim()[-1])
# Axis labels
axsacmer.set_xlabel('t [ms]')
axsacmer.set_ylabel('Absolute velocity [°/s]')
# Legend etc.
axsacmer.legend(loc='upper right')
# Save figure
figsacmer.subplots_adjust(top=0.93, bottom=0.135, left=0.170, right=0.97, hspace=0.2, wspace=0.2)
plt.pause(0.001)
[figsacmer.savefig(os.path.join(cfg.fsd, 'zebrafish/saccade_velocities%s' % ext)) for ext in cfg.exts]
plt.close(figsacmer)

# Figure 6 Noise distribution boxplots
linewidth = 2.5
figbox, axbox = plt.subplots(1, 2, figsize=(10.91, 6.6))

# Do in the first subplots the multiplicative errors.
s_epszf = np.concatenate([zfish_data.s_epstns, zfish_data.s_epsnts])
s_epsmac = np.concatenate([mac_data.s_epsdlall, mac_data.s_epsulall, mac_data.s_epsdrall, mac_data.s_epsurall])

box = axbox[0].boxplot([s_epszf[~np.isnan(s_epszf)], s_epsmac[~np.isnan(s_epsmac)]], patch_artist=True, widths=0.75)
# Define outline colors
outline_colors = ['blue', 'red']
# Set box outline colors and transparent fill
for patch, color in zip(box['boxes'], outline_colors):
    patch.set_facecolor('none')
    patch.set_edgecolor(color)
# Color whiskers, caps, medians, and fliers
for i in range(2):
    color = outline_colors[i]
    box['whiskers'][2*i].set_color(color)
    box['whiskers'][2*i + 1].set_color(color)
    box['caps'][2*i].set_color(color)
    box['caps'][2*i + 1].set_color(color)
    box['medians'][i].set_color(color)
    box['fliers'][i].set_markeredgecolor(color)

    # Set linewidths
    box['whiskers'][2*i].set_linewidth(linewidth)
    box['whiskers'][2*i + 1].set_linewidth(linewidth)
    box['caps'][2*i].set_linewidth(linewidth)
    box['caps'][2*i + 1].set_linewidth(linewidth)
    box['medians'][i].set_linewidth(linewidth)
for patch in box['boxes']:
    patch.set_linewidth(linewidth)

axbox[0].set_xticklabels(['Zebrafish', 'Macaque'])
axbox[0].set_yscale('log')
axbox[0].set_ylabel(r's$_\epsilon$')

# Now do the bars
xizf =  np.array([a for b in [zfish_data.xitn, zfish_data.xint] for a in b])
ximac = np.array([a for b in [mac_data.xidl,mac_data.xidr, mac_data.xiul, mac_data.xiur] for a in b])
# Percentile
xizf = xizf[(xizf < np.percentile(xizf, 99.5)) & (xizf > np.percentile(xizf, 0.5))]
ximax = ximac[(ximac < np.percentile(ximac, 99.5)) & (ximac > np.percentile(ximac, 0.5))]

se_sds = [sd / np.sqrt(2 * (n - 1)) for sd, n in zip([zfish_data.s_xipooledpercentiled, mac_data.sxipooled[-1]], [xizf.shape[0], ximac.shape[0]])]
colors = ['blue', 'red']
bar_width = 0.6
# Create bars with no error bars (we'll add them manually)
bars = axbox[1].bar([0,1], [zfish_data.s_xipooledpercentiled, mac_data.sxipooled[-1]], width=bar_width, color=colors, edgecolor=colors, linewidth=linewidth)
# Add custom-colored error bars (SE of SD)
for i, (xpos, mean, err) in enumerate(zip([0,1], [zfish_data.s_xipooledpercentiled, mac_data.sxipooled[-1]], se_sds)):
    axbox[1].errorbar(x=xpos, y=mean, yerr=err, fmt='none',
                ecolor='k', elinewidth=linewidth, capsize=8, capthick=linewidth)

axbox[1].set_xticks([0, 1])
axbox[1].set_xticklabels(['Zebrafish', 'Macaque'])
axbox[1].set_ylabel(r'$s_\xi$')
figbox.subplots_adjust(top=0.975, bottom=0.11, left=0.125, right=0.97, hspace=0.2, wspace=0.39)
# Save
plt.pause(0.001)
[figbox.savefig(os.path.join(cfg.fsd, 'joint/noise_distribution_boxplots%s' % ext)) for ext in cfg.exts]
plt.close(figbox)

# Figure 7 removed saccades zfish
# One figure showing removed saccades, possibly 2 with left out saccades right noisy saccades
figout, axout = plt.subplots(1,2, sharex=True, sharey=True)
plt.get_current_fig_manager().window.showMaximized()
figout.subplots_adjust(wspace=0.06)
axout[0].plot(np.arange(0,1001), zfish_data.saccadedataout[:,0,:]-np.nanmean(zfish_data.saccadedataout[:50,0,:], axis=0), 'k-', alpha=0.5)
axout[1].plot(np.arange(0,1001), zfish_data.saccadedataout[:,1,:]-np.nanmean(zfish_data.saccadedataout[:50,1,:], axis=0), 'k-', alpha=0.5)
axout[0].plot(np.arange(0,1022), zfish_data.saccadedatanoise[:,0,:]-np.nanmean(zfish_data.saccadedatanoise[:50,0,:], axis=0), 'k-', alpha=0.5)
axout[1].plot(np.arange(0,1022), zfish_data.saccadedatanoise[:,1,:]-np.nanmean(zfish_data.saccadedatanoise[:50,1,:], axis=0), 'k-', alpha=0.5)
# Adjust
axout[0].set_ylabel('Eye position [°]')
axout[0].set_xlabel('t [ms]', x=1)
for j, tit2 in enumerate(['Left', 'Right']):
    axout[j].set_title(f'{tit2} eye')
plt.pause(0.001)
[figout.savefig(os.path.join(cfg.fsd, 'zebrafish/removed_saccades%s' % ext)) for ext in cfg.exts]
plt.close(figout)

# Figure for epsilon vs u macaque and zebrafish
figepsu, axepsu = plt.subplots(1, 2, figsize=(10.91, 6.6))
axepsu[0].scatter(np.concatenate([zfish_data.ubinstn[zfish_data.ubinstn>cfg.ucut], zfish_data.ubinsnt[zfish_data.ubinsnt>cfg.ucut]]),
                  np.concatenate([zfish_data.s_epstns, zfish_data.s_epsnts]), color='blue', alpha=0.5)
axepsu[1].scatter(np.concatenate([mac_data.udl[mac_data.udl>cfg.ucut][~np.isnan(mac_data.s_epsdlall)], mac_data.uul[mac_data.uul>cfg.ucut], mac_data.udr[mac_data.udr>cfg.ucut], mac_data.uur[mac_data.uur>cfg.ucut]]),
                  np.concatenate([mac_data.s_epsdlall[~np.isnan(mac_data.s_epsdlall)], mac_data.s_epsulall, mac_data.s_epsdrall, mac_data.s_epsurall]), color='red', alpha=0.5)
# Adjust figure
axepsu[0].set_xlabel('u [°/s]', x=1)
axepsu[0].set_ylabel(r's$_\epsilon$ [°]')
axepsu[0].set_title('Zebrafish')
axepsu[1].set_title('Macaque')
figepsu.subplots_adjust(top=0.89, bottom=0.184, left=0.123, right=0.966, hspace=0.2, wspace=0.254)
# Save figure
plt.pause(0.001)
[figepsu.savefig(os.path.join(cfg.fsd, 'joint/command_signal_noise%s' % ext)) for ext in cfg.exts]
plt.close(figepsu)

# Figure 7 showing how fit residual depends on the control signal u for zebrafish and macaque
figresu, axresu = plt.subplots(2, 1, figsize=(9.3, 5.7), sharex=True)
figresu.subplots_adjust(top=0.98, bottom=0.16, left=0.175, right=0.985, hspace=0.2, wspace=0.2)
# Plot
# Zebrafish
axresu[0].scatter(np.concatenate([zfish_data.ubinstn[zfish_data.ubinstn>cfg.ucut], zfish_data.ubinsnt[zfish_data.ubinsnt>cfg.ucut]]),
                  np.concatenate([zfish_data.stdnperbintn[zfish_data.ubinstn>cfg.ucut], zfish_data.stdnperbinnt[zfish_data.ubinsnt>cfg.ucut]]), color='blue', alpha=0.5)
# Macaque
axresu[1].scatter(np.concatenate([mac_data.udl[mac_data.udl>cfg.ucut], mac_data.uul[mac_data.uul>cfg.ucut], mac_data.udr[mac_data.udr>cfg.ucut], mac_data.uur[mac_data.uur>cfg.ucut]]),
                  np.concatenate([mac_data.nbdl[mac_data.udl>cfg.ucut], mac_data.nbul[mac_data.uul>cfg.ucut], mac_data.nbdr[mac_data.udr>cfg.ucut], mac_data.nbur[mac_data.uur>cfg.ucut]]), color='red', alpha=0.5)

# 'Plot; for the legend
axresu[0].scatter([], [], color='blue', alpha=0.75, label='Zebrafish')
axresu[0].scatter([], [], color='red', alpha=0.75, label='Macaque')

# Adjustments
axresu[0].legend(loc='upper right')
axresu[1].set_xlabel(r'$\overline{u}$')
axresu[1].set_ylabel(r'Fit error', y=1)
# Save
plt.pause(0.001)
[figresu.savefig(os.path.join(cfg.fsd, 'joint/command_signal_residual%s' % ext)) for ext in cfg.exts]
plt.close(figresu)

# Figure 8 violin plots for each animal for the additive and multiplicative noises
fignperan, axnperan = plt.subplots(2, 1, figsize=(8.3, 5.7), sharex=True)
fignperan.subplots_adjust(top=0.945, bottom=0.095, left=0.17, right=0.975, hspace=0.1, wspace=0.19)
for ax in axnperan:
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yscale('log')

# Violin plots: Sort by sample size and write the number of samples above the violin plot
n_saccades_per = np.array([len(s) for s in zfish_data.s_xi_per_animal])
n_sortidx = np.argsort(n_saccades_per)
parts_xi = axnperan[0].violinplot(np.array(zfish_data.s_xi_per_animal, dtype=object)[n_sortidx], showmeans=False, showmedians=True, showextrema=True)
eps_per_animal = [[np.nan]*2 if len(_) == 0 else _ for _ in zfish_data.eps_per_animal]  # Adjust empty lists
parts_eps = axnperan[1].violinplot(np.array(eps_per_animal, dtype=object)[n_sortidx], showmeans=False, showmedians=True, showextrema=True)

# Add the macaque counterpart as dashed line
xlim = axnperan[0].get_xlim()
axnperan[0].plot(xlim, [mac_data.sxipooled[0]]*2, 'k--', zorder=3, lw=2)
axnperan[1].plot(xlim, [s_epsmac.mean()]*2, 'k--', zorder=3, lw=2)

axnperan[0].set_xlim(xlim)

# Add the sample size as text above each violin plot
for i, n in enumerate(n_saccades_per[n_sortidx]):
    axnperan[0].text(i+1, axnperan[0].get_ylim()[1]*1.5, f'{n}', ha='center', va='top', fontsize=25)

# Violinplots adjustment
for part in [parts_xi, parts_eps]:
    for pc in part["bodies"]:
        pc.set_facecolor("#2b7bba")  # Clean uniform blue
        pc.set_edgecolor("#1a4d75")  # Slightly darker blue border
        pc.set_alpha(0.85)  # Crisper opacity

    part["cmedians"].set_edgecolor("black")  # Bold black for median
    part["cmedians"].set_linewidth(2)
    part["cmaxes"].set_edgecolor("black")  # Dark gray for min/max limits
    part["cmins"].set_edgecolor("black")
    part["cbars"].set_edgecolor("black")  # Center backbone connecting min

axnperan[1].set_xlabel('Fish ID')
axnperan[0].set_ylabel(r'$s_{\xi}$')
axnperan[1].set_ylabel(r'$s_{\epsilon}$')

# Save
plt.pause(0.001)
[fignperan.savefig(os.path.join(cfg.fsd, 'zebrafish/noise_per_animal%s' % ext)) for ext in cfg.exts]
plt.close(fignperan)