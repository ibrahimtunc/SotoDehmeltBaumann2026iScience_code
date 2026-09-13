#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import function_scripts as fnc
import config as cfg
import os
from types import SimpleNamespace
from scipy.stats import kruskal, mannwhitneyu
import pandas as pd
from scipy.optimize import curve_fit

# Estimate the additive & multiplicative noise given zebrafish and macaque datasets.

# Load data and pack it to a SimpleNamespace for easier access
with np.load(os.path.join(cfg.save_path, 'macaque_saccade_data_processed.npz'), allow_pickle=True) as loader_mac:
    mac_data = SimpleNamespace(**loader_mac)
with np.load(os.path.join(cfg.save_path, 'zebrafish_saccade_data_processed.npz'), allow_pickle=True) as loader_zfish:
    zfish_data = SimpleNamespace(**loader_zfish)

# MACAQUE
print('Motor noise estimation for macaque')
# Motor noise
# Additive xi:
print('Estimating additive noise (xi)')
#down left
xidl, prenumdl, postnumdl, sxidl = fnc.macaque_xi_estimation(mac_data.dl, mac_data.onsetdl, mac_data.offsetdl)
#down right
xidr, prenumdr, postnumdr, sxidr = fnc.macaque_xi_estimation(mac_data.dr, mac_data.onsetdr, mac_data.offsetdr)
#up left
xiul, prenumul, postnumul, sxiul = fnc.macaque_xi_estimation(mac_data.ul, mac_data.onsetul, mac_data.offsetul)
#down right
xiur, prenumur, postnumur, sxiur = fnc.macaque_xi_estimation(mac_data.ur, mac_data.onsetur, mac_data.offsetur)

#pool all xi together
xipooled = np.array([a for b in [xidl,xidr, xiul, xiur] for a in b])
xipooled = xipooled[~np.isnan(xipooled)]
#take 99th percentile -> remove first 0.5 (negative outliers) and last 0.5 (positive outliers)
xipperc = xipooled[(xipooled<np.percentile(xipooled, 99.5)) & (xipooled>np.percentile(xipooled, 0.5))]
sxipooled = fnc.calculate_s_xi(xipooled)

# Multiplicative epsilon:
print('Estimating multiplicative noise (epsilon)')
#Motor noise 2: epsilon -> trace during saccade
s_epsdl, udl, s_epsdlall, tndl, nbdl, nsidl, restdl = fnc.macaque_eps_estimation(mac_data.dl, xipperc, mac_data.onsetdl, mac_data.offsetdl,
                                                 pre_post_dur=cfg.pre_post_dur, return_RMS=True, return_fitparams=True)

s_epsdr, udr, s_epsdrall, tndr, nbdr, nsidr, restdr = fnc.macaque_eps_estimation(mac_data.dr, xipperc, mac_data.onsetdr, mac_data.offsetdr,
                                                 pre_post_dur=cfg.pre_post_dur, return_RMS=True, return_fitparams=True)

s_epsul, uul, s_epsulall, tnul, nbul, nsiul, restul = fnc.macaque_eps_estimation(mac_data.ul, xipperc, mac_data.onsetul, mac_data.offsetul,
                                                 pre_post_dur=cfg.pre_post_dur, return_RMS=True, return_fitparams=True)

s_epsur, uur, s_epsurall, tnur, nbur, nsiur, restur = fnc.macaque_eps_estimation(mac_data.ur, xipperc, mac_data.onsetur, mac_data.offsetur,
                                                 pre_post_dur=cfg.pre_post_dur, return_RMS=True, return_fitparams=True)

perc_used_mac = ((nsidl.shape[0] + nsidr.shape[0] + nsiul.shape[0] + nsiur.shape[0]) /
                 (mac_data.dl.shape[0] + mac_data.dr.shape[0] + mac_data.ul.shape[0] + mac_data.ur.shape[0]))
print('%% %.2f used' % (perc_used_mac*100))

print('Difference between epsilon estimates (ANOVA)')
print(kruskal(s_epsdlall, s_epsdrall, s_epsulall, s_epsurall))


# ZEBRAFISH
print('Motor noise estimation for zebrafish')
# Motor noise 1: xi -> fit a linear curve to nonsaccadic traces, find the deviation from the fit and calculate s_xi
print('Estimating additive noise (xi)')
# nasal->temporal
xint = []  # xi nasal to temporal
prenumnt = 0  # number of presaccadic traces used in the analysis
postnumnt = 0  # number of postsaccadic traces used in the analysis
for idx, trace in enumerate(zfish_data.tracesnt):
    presaccade = trace[:zfish_data.onsetnt[idx]]
    postsaccade = trace[zfish_data.offsetnt[idx]:]

    if len(presaccade) < 20:  # if less than 20 ms available for presaccade, discard the presaccadic curve from analysis
        presaccade = np.empty(0)
        prefit = presaccade.copy()
    else:
        prenumnt += 1
        presaccade = trace[5:zfish_data.onsetnt[idx] - 5]  # discard 5 ms from beginning and end of presaccadic fixation
        xpre = np.arange(0, len(presaccade))
        preparams, _ = curve_fit(fnc.linear_fit, xpre, presaccade)
        prefit = fnc.linear_fit(xpre, *preparams)

    if len(postsaccade) < 20:  # same story
        postsaccade = np.empty(0)
        postfit = postsaccade.copy()
    else:
        postnumnt += 1
        postsaccade = trace[zfish_data.offsetnt[idx] + 5:-5]  # discard again
        xpost = np.arange(0, len(postsaccade))
        postparams, _ = curve_fit(fnc.linear_fit, xpost, postsaccade)
        postfit = fnc.linear_fit(xpost, *postparams)

    xint.append(np.concatenate((presaccade - prefit, postsaccade - postfit)))

xint_per_saccade = xint.copy()
xint = np.array([a for b in xint for a in b])
# find outliers
xintiqr = np.percentile(xint, 75) - np.percentile(xint, 25)
xintout = xint[(xint < np.median(xint) - 1.5 * xintiqr) | (xint > np.median(xint) + 1.5 * xintiqr)]
xintfiltered = xint[(xint > np.median(xint) - 1.5 * xintiqr) & (xint < np.median(xint) + 1.5 * xintiqr)]
s_xintraw = np.std(xint)
# outliers as 1.5 IQR
s_xintfiltered = np.std(xintfiltered)
# outliers as above 99th percentile
s_xintpercentiled = np.std(xint[xint < np.percentile(xint, 99)])


# temporal->nasal
xitn = []  # xi temporal to nasal
prenumtn = 0
postnumtn = 0
for idx, trace in enumerate(zfish_data.tracestn):
    presaccade = trace[:zfish_data.onsettn[idx]]
    postsaccade = trace[zfish_data.offsettn[idx]:]

    if len(presaccade) < 20:  # if less than 20 ms available for presaccade, discard the presaccadic curve from analysis
        presaccade = np.empty(0)
        prefit = presaccade.copy()
    else:
        prenumtn += 1
        presaccade = trace[5:zfish_data.onsettn[idx] - 5]  # discard 5 ms from beginning and end of presaccadic fixation
        xpre = np.arange(0, len(presaccade))
        preparams, _ = curve_fit(fnc.linear_fit, xpre, presaccade)
        prefit = fnc.linear_fit(xpre, *preparams)

    if len(postsaccade) < 20:  # same story
        postsaccade = np.empty(0)
        postfit = postsaccade.copy()
    else:
        postnumtn += 1
        postsaccade = trace[zfish_data.offsettn[idx] + 5:-5]  # discard again
        xpost = np.arange(0, len(postsaccade))
        postparams, _ = curve_fit(fnc.linear_fit, xpost, postsaccade)
        postfit = fnc.linear_fit(xpost, *postparams)

    xitn.append(np.concatenate((presaccade - prefit, postsaccade - postfit)))

xitn_per_saccade = xitn.copy()

xitn = np.array([a for b in xitn for a in b])
xitn = xitn[~np.isnan(xitn)]
# find outliers
xitniqr = np.percentile(xitn, 75) - np.percentile(xitn, 25)
xitnout = xitn[(xitn < np.median(xitn) - 1.5 * xitniqr) | (xitn > np.median(xitn) + 1.5 * xitniqr)]
xitnfiltered = xitn[(xitn > np.median(xitn) - 1.5 * xitniqr) & (xitn < np.median(xitn) + 1.5 * xitniqr)]
s_xitnraw = np.std(xitn)  # 0.686° smaller after removing data points from beginning and right before saccade
# outliers as 1.5 IQR
s_xitnfiltered = np.std(xitnfiltered)  # 0.0101 slightly bigger (0.001)
# outliers as above 99th percentile (probably this is the safest way to go)
s_xitnpercentiled = np.std(xitn[xitn < np.percentile(xitn, 99)])  # 0.0386 slightly smaller (0.0001)

# see how s_xi is when all traces are pooled
xipooled = np.array([a for b in [xitn, xint] for a in b])
# find outliers
xipoolediqr = np.percentile(xipooled, 75) - np.percentile(xipooled, 25)
xipooledout = xipooled[(xipooled < np.median(xipooled) - 1.5 * xipoolediqr) | (xipooled > np.median(xipooled) + 1.5 * xipoolediqr)]
xipooledfiltered = xipooled[(xipooled > np.median(xipooled) - 1.5 * xipoolediqr) & (xipooled < np.median(xipooled) + 1.5 * xipoolediqr)]
s_xipooledraw = np.std(xipooled)
# outliers as 1.5 IQR
s_xipooledfiltered = np.std(xipooledfiltered)
# outliers as above 99th percentile (probably this is the safest way to go)
s_xipooledpercentiled = np.std(xipooled[(xipooled < np.percentile(xipooled, 99.5)) &
                                        (xipooled > np.percentile(xipooled, 0.5))])

# since values (especially outlier filtered) are similar, it is plausible to use pooled data.
# Estimates per saccade
# t - n
s_xitn_per_saccade = np.array([np.std(x) for x in xitn_per_saccade])
s_xitn_per_saccade_percentiled = np.array([np.std((x[(x < np.percentile(xipooled, 99.5)) & (x > np.percentile(xipooled, 0.5))]))
                                           for x in xitn_per_saccade])
# n - t
s_xint_per_saccade = np.array([np.std(x) for x in xint_per_saccade])
s_xint_per_saccade_percentiled = np.array([np.std((x[(x < np.percentile(xipooled, 99.5)) & (x > np.percentile(xipooled, 0.5))]))
                                           for x in xint_per_saccade])

# GROUP THE s_xi (nt / tn) PER ANIMAL to get the distributions
unique_ids = list(set(zfish_data.id_tn) & set(zfish_data.id_nt))
s_xi_per_animal = len(unique_ids) * [np.nan]
s_xi_per_animal_percentiled = len(unique_ids) * [np.nan]
for i, id in enumerate(unique_ids):
    print(id)
    s_xi_per_animal[i] = np.concatenate([s_xint_per_saccade[zfish_data.id_nt == id], s_xitn_per_saccade[zfish_data.id_tn == id]])
    s_xi_per_animal_percentiled[i] = np.concatenate([s_xint_per_saccade_percentiled[zfish_data.id_nt == id], s_xitn_per_saccade_percentiled[zfish_data.id_tn == id]])

# Motor noise 2: epsilon -> trace during saccade

# temporal->nasal
ntn = []  # temporal-nasal deviation of eye traces from saccade fit
utn = []
RMSstn = np.zeros(len(zfish_data.tracestn))
fitparamstn = []
for idx, trace in enumerate(zfish_data.tracestn):
    preidx = zfish_data.onsettn[idx] - cfg.pre_post_dur
    postidx = zfish_data.offsettn[idx] + cfg.pre_post_dur
    if preidx < 0:
        preidx = 0
    if postidx > trace.shape[0]:
        postidx = trace.shape[0]
    saccade = trace[preidx:postidx]
    t = np.arange(0, len(saccade))
    fitparams, _ = curve_fit(fnc.saccade_fit_func, t, saccade, method='lm', maxfev=100000)
    fittedsac = fnc.saccade_fit_func(t, *fitparams)
    # NOTE THAT 2-3 CASES FIT FLAT LINE! Discard them since u will be zero
    totnoise = (saccade - fittedsac)
    totnoise = totnoise[1:] # Align noise with u values
    utemp = np.abs(np.diff(fittedsac))  # template u
    ntn.append(totnoise)
    utn.append(utemp)
    RMSstn[idx] = np.sqrt(np.mean(totnoise ** 2))
    fitparamstn.append(fitparams)
    # print(idx)
nicesacidxstn = np.where(RMSstn <= np.mean(RMSstn) - cfg.RMSfac * np.std(RMSstn))[0]

# Get the noise and derivatives in a per-saccade manner to be used later down the line
ntn_per_saccade = ntn.copy()
utn_per_saccade = utn.copy()
# Pool over all saccades and sort by u values
ntn = np.concatenate([ntn[i] for i in nicesacidxstn])
utn = np.concatenate([utn[i] for i in nicesacidxstn])
(utn, ntn) = zip(*sorted(zip(utn, ntn)))  # sort u and noise arrays in ascending u order

# bin u values and epsilon values -> 03.05.2022 : Bin it per percentile, that you take first 100 smallest u values
# and bin is the average value of these 100 values. -> This is almost like logarithmic.
# Alternative idea is logarithmic binning of u
ubinstn = []
binnedutn = []  # u bins
binnedntn = []  # total noise bins

for k in range(np.ceil(len(utn) / cfg.nperbin).astype(int)):
    # print(k * nperbin, k * nperbin + nperbin)
    if k * cfg.nperbin + cfg.nperbin < len(utn):
        us = utn[k * cfg.nperbin: k * cfg.nperbin + cfg.nperbin]
        ns = ntn[k * cfg.nperbin: k * cfg.nperbin + cfg.nperbin]
    else:
        us = utn[k * cfg.nperbin:]
        ns = ntn[k * cfg.nperbin:]
    ubinstn.append(np.mean(us))
    binnedutn.append(us)
    binnedntn.append(ns)

ubinstn = np.array(ubinstn)

# find average s_eps per bin and then average over all
usidx = np.where(ubinstn > cfg.ucut)[0][0]  # the first u bin index value bigger than cutoff

stdnperbintn = np.array([np.std(a) for a in binnedntn])  # standard deviation of total noise
s_epstns = 1 / np.abs(ubinstn[usidx:]) * np.sqrt(stdnperbintn[usidx:] ** 2 - s_xipooledpercentiled ** 2)
s_epstn = np.mean(s_epstns[~np.isnan(s_epstns)])  # 1.824 as of 4.3.2022, 7.818 (!) after binning u according to percentiles. CHECKS OUT AFTER 2025

# nasal->temporal
nnt = []
unt = []
RMSsnt = np.zeros(len(zfish_data.tracesnt))
fitparamsnt = []
for idx, trace in enumerate(zfish_data.tracesnt):
    preidx = zfish_data.onsetnt[idx] - cfg.pre_post_dur
    postidx = zfish_data.offsetnt[idx] + cfg.pre_post_dur
    if preidx < 0:
        preidx = 0
    if postidx > trace.shape[0]:
        postidx = trace.shape[0]
    saccade = trace[preidx:postidx]
    t = np.arange(0, len(saccade))
    fitparams, _ = curve_fit(fnc.saccade_fit_func, t, saccade, method='lm', maxfev=100000)
    fittedsac = fnc.saccade_fit_func(t, *fitparams)
    # NOTE THAT 2-3 CASES FIT A FLAT LINE! Discard them since u will be zero
    totnoise = (saccade - fittedsac)  # /np.sqrt(dt)
    totnoise = totnoise[1:]
    utemp = np.abs(np.diff(fittedsac))  # template u
    nnt.append(totnoise)
    unt.append(utemp)
    RMSsnt[idx] = np.sqrt(np.mean(totnoise**2))
    fitparamsnt.append(fitparams)

rmsidxnt = ~np.isnan(RMSsnt)
nicesacidxsnt = np.where(RMSsnt <= np.mean(RMSsnt[rmsidxnt]) - cfg.RMSfac * np.std(RMSsnt[rmsidxnt]))[0]
badsacidxsnt = np.where(RMSsnt > np.mean(RMSsnt[rmsidxnt]) - cfg.RMSfac * np.std(RMSsnt[rmsidxnt]))[0]

# I will again use these estimates later down the line.
nnt_per_saccade = nnt.copy()
unt_per_saccade = unt.copy()

nnt = np.concatenate([nnt[i] for i in nicesacidxsnt])
unt = np.concatenate([unt[i] for i in nicesacidxsnt])
(unt, nnt) = zip(*sorted(zip(unt, nnt)))  # sort u and noise arrays in ascending u order

ubinsnt = []
binnedunt = []  # u bins
binnednnt = []  # total noise bins

for k in range(np.ceil(len(unt) / cfg.nperbin).astype(int)):
    # print(k * nperbin, k * nperbin + nperbin)
    if k * cfg.nperbin + cfg.nperbin < len(unt):
        us = unt[k * cfg.nperbin: k * cfg.nperbin + cfg.nperbin]
        ns = nnt[k * cfg.nperbin: k * cfg.nperbin + cfg.nperbin]
    else:
        us = unt[k * cfg.nperbin:]
        ns = nnt[k * cfg.nperbin:]
    ubinsnt.append(np.mean(us))
    binnedunt.append(us)
    binnednnt.append(ns)

ubinsnt = np.array(ubinsnt)

# find average s_eps per bin and then average over all
usidx = np.where(ubinsnt > cfg.ucut)[0][0]  # the first u bin index value bigger than cutoff

stdnperbinnt = np.array([np.std(a) for a in binnednnt])  # standard deviation of total noise
s_epsnts = 1 / np.abs(ubinsnt[usidx:]) * np.sqrt(stdnperbinnt[usidx:] ** 2 - s_xipooledpercentiled ** 2)
s_epsnt = np.mean(s_epsnts[~np.isnan(s_epsnts)])  # 1.824 as of 4.3.2022, 10.208(!) as of 03.05 after percentile binning. SOMEHOW 2025 8.145!

perc_used_zf = (nicesacidxstn.shape[0] + nicesacidxsnt.shape[0]) / (len(zfish_data.tracestn) + len(zfish_data.tracesnt))
print('%% %.2f used' % (perc_used_zf * 100))

# Finally, do the per individual case, pool per animal everything on n and u, sort them and bin them accordingly to estimate
# Use the same first 100 values for the per-animal binning to keep things comparable.
# Convert to arrays
nnt_per_saccade = np.array(nnt_per_saccade, dtype=object)
unt_per_saccade = np.array(unt_per_saccade, dtype=object)
ntn_per_saccade = np.array(ntn_per_saccade, dtype=object)
utn_per_saccade = np.array(utn_per_saccade, dtype=object)

eps_per_animal = len(unique_ids) * [np.nan] # Preallocated array

for i, id in enumerate(unique_ids):
    print(id)
    # ADD THE EXCLUSION BACK!
    rmsnt = RMSsnt[zfish_data.id_nt == id]
    rmstn = RMSstn[zfish_data.id_tn == id]
    nicesacidxsnt = np.where(rmsnt <= np.mean(RMSsnt[rmsidxnt]) - cfg.RMSfac * np.std(RMSsnt[rmsidxnt]))[0]
    nicesacidxstn = np.where(rmstn <= np.mean(RMSstn) - cfg.RMSfac * np.std(RMSstn))[0]
    try:
        n_per_animal = np.concatenate(np.concatenate([nnt_per_saccade[nicesacidxsnt], ntn_per_saccade[nicesacidxstn]]))
        u_per_animal = np.concatenate(np.concatenate([unt_per_saccade[nicesacidxsnt], utn_per_saccade[nicesacidxstn]]))
    except ValueError:
        print('No saccades for animal %s after RMS filtering, skipping...' % id)
        eps_per_animal[i] = []
        continue
    # print(n_per_animal.shape, u_per_animal.shape)

    # Sort by u per animal
    (u_per_animal, n_per_animal) = zip(*sorted(zip(u_per_animal, n_per_animal)))  # sort u and noise arrays in ascending u order

    ubins = []
    binnedu = []  # u bins
    binnedn = []  # total noise bins

    for k in range(np.ceil(len(u_per_animal) / cfg.nperbin).astype(int)):
        if k * cfg.nperbin + cfg.nperbin < len(u_per_animal):
            us = u_per_animal[k * cfg.nperbin: k * cfg.nperbin + cfg.nperbin]
            ns = n_per_animal[k * cfg.nperbin: k * cfg.nperbin + cfg.nperbin]
        else:
            us = u_per_animal[k * cfg.nperbin:]
            ns = n_per_animal[k * cfg.nperbin:]
        ubins.append(np.mean(us))
        binnedu.append(us)
        binnedn.append(ns)

    ubins = np.array(ubins)

    # find average s_eps per bin and then average over all
    usidx = np.where(ubins > cfg.ucut)[0][0]  # the first u bin index value bigger than cutoff

    stdnperbin = np.array([np.std(a) for a in binnedn])  # standard deviation of total noise
    s_epss = 1 / np.abs(ubins[usidx:]) * np.sqrt(stdnperbin[usidx:] ** 2 - s_xipooledpercentiled ** 2)
    eps_per_animal[i] = s_epss

print('Epsilon estimate differences between groups')
print(mannwhitneyu(s_epsnts, s_epstns)) # They are not normal according to shapiro wilkins test (scipt.stats.shapiro)


# Save the estimates generated
print('Saving results...')
# Arrays
# a) Macaque
np.savez(os.path.join(cfg.save_path, 'macaque_noise_estimates.npz'),
         xidl=xidl, xidr=xidr, xiul=xiul, xiur=xiur, xipooled=xipooled, sxipooled=sxipooled,
         s_epsdl=s_epsdl, s_epsdr=s_epsdr, s_epsul=s_epsul, s_epsur=s_epsur,
         s_epsdlall=s_epsdlall, s_epsdrall=s_epsdrall, s_epsulall=s_epsulall, s_epsurall=s_epsurall,
         udl=udl, udr=udr, uul=uul, uur=uur,
         tndl=tndl, tndr=tndr, tnul=tnul, tnur=tnur,
         nbdl=nbdl, nbdr=nbdr, nbul=nbul, nbur=nbur,
         nsidl=nsidl, nsidr=nsidr, nsiul=nsiul, nsiur=nsiur,
         restdl=restdl, restdr=restdr, restul=restul, restur=restur)
# b) Zebrafish
np.savez(os.path.join(cfg.save_path, 'zebrafish_noise_estimates.npz'),
         xint=xint, xitn=xitn, xipooled=xipooled, s_xipooledpercentiled=s_xipooledpercentiled,
         RMSstn=RMSstn, RMSsnt=RMSsnt, fitparamstn=fitparamstn, fitparamsnt=fitparamsnt,
         s_epsnts=s_epsnts, s_epstns=s_epstns, s_epsnt=s_epsnt, s_epstn=s_epstn, ubinsnt=ubinsnt, ubinstn=ubinstn,
         binnednnt=np.array(binnednnt, dtype=object), binnedntn=np.array(binnedntn, dtype=object),
         binnedunt=np.array(binnedunt, dtype=object), binnedutn=np.array(binnedutn, dtype=object),
         nnt_per_saccade=nnt_per_saccade, ntn_per_saccade=ntn_per_saccade,
         unt_per_saccade=unt_per_saccade, utn_per_saccade=utn_per_saccade,
         s_xi_per_animal=np.array(s_xi_per_animal, dtype=object), s_xi_per_animal_percentiled=np.array(s_xi_per_animal_percentiled, dtype=object),
         eps_per_animal=np.array(eps_per_animal, dtype=object), unique_ids=unique_ids, stdnperbinnt=stdnperbinnt, stdnperbintn=stdnperbintn)

# Data used for plotting & other purposes
# SAVE DATA
# ----------------

# Save the noise estimates
# A) ADDITIVE
# a) macaque
macxi = np.array([a for b in [xidl,xidr, xiul, xiur] for a in b])
macxi = macxi[(macxi<np.percentile(macxi, 99.5)) & (macxi>np.percentile(macxi, 0.5))]
# b) zebrafish
zfxi = np.array([a for b in [xitn, xint] for a in b])
zfxi = zfxi[(zfxi<np.percentile(zfxi, 99.5)) & (zfxi>np.percentile(zfxi, 0.5))]

# Create dataframe
dfxi = pd.DataFrame({'animal': ['macaque']*len(macxi) + ['zebrafish']*len(zfxi),
                     'xi': np.concatenate([macxi, zfxi])})
# Save dataframe
dfxi.to_csv(os.path.join(cfg.save_path, 'xi.csv'), index=False)

# c) zebrafish but per animal
dfxi_zf_per_animal = pd.DataFrame({'fish_ID': np.repeat(unique_ids, [len(a) for a in s_xi_per_animal]),
                                   's_xi_raw': np.concatenate(s_xi_per_animal),
                                   's_xi_percentiled': np.concatenate(s_xi_per_animal_percentiled)})

# Save dataframe
dfxi_zf_per_animal.to_csv(os.path.join(cfg.save_path, 's_xi_zf_per_animal.csv'), index=False)

# B) MULTIPLICATIVE
# a) macaque
s_epsmac = np.concatenate([s_epsdlall, s_epsulall, s_epsdrall, s_epsurall])
usmac = np.concatenate([udl[udl>cfg.ucut], uul[uul>cfg.ucut], udr[udr>cfg.ucut], uur[uur>cfg.ucut]])
dirsmac = np.repeat(['down_left', 'up_left', 'down_right', 'up_right'],
                    [len(udl[udl>cfg.ucut]), len(uul[uul>cfg.ucut]), len(udr[udr>cfg.ucut]), len(uur[uur>cfg.ucut])])
# REMOVE NANs
usmac = usmac[~np.isnan(s_epsmac)]
dirsmac = dirsmac[~np.isnan(s_epsmac)]
s_epsmac = s_epsmac[~np.isnan(s_epsmac)]

# b) zebrafish
s_epszf = np.concatenate([s_epstns, s_epsnts])
uszf = np.concatenate([ubinstn[ubinstn>cfg.ucut], ubinsnt[ubinsnt>cfg.ucut]])
dirszf = np.repeat(['temporal_nasal', 'nasal_temporal'], [len(ubinstn[ubinstn>cfg.ucut]), len(ubinsnt[ubinsnt>cfg.ucut])])

# REMOVE NANs
uszf = uszf[~np.isnan(s_epszf)]
dirszf = dirszf[~np.isnan(s_epszf)]
s_epszf = s_epszf[~np.isnan(s_epszf)]

# Create the dataframe
dfeps = pd.DataFrame({'animal' : ['macaque']*len(s_epsmac) + ['zebrafish']*len(s_epszf),
                     's_eps' : np.concatenate([s_epsmac, s_epszf]),
                      'u' : np.concatenate([usmac, uszf]),
                      'direction' : np.concatenate([dirsmac, dirszf])})
dfeps.to_csv(os.path.join(cfg.save_path, 's_eps.csv'), index=False)

# c) zebrafish but per animal
dfeps_zf_per_animal = pd.DataFrame({'fish_ID': np.repeat(unique_ids, [len(a) for a in eps_per_animal]),
                                    's_eps': np.concatenate(eps_per_animal)})

dfeps_zf_per_animal.to_csv(os.path.join(cfg.save_path, 's_eps_zf_per_animal.csv'), index=False)


# 2) Dataframes showing the noise estimates (i.e. the ones used for box plots)
# A) ADDITIVE
dfs_xi = pd.DataFrame({'animal' : ['macaque', 'zebrafish'], 's_xi' : [sxipooled[-1], s_xipooledpercentiled]})
dfs_xi.to_csv(os.path.join(cfg.save_path, 's_xi.csv'), index=False)

# Get the number of saccades per animal & what is removed / considered
# FIND THE TOTAL NUMBER OF ADDED / DISCARDED SACCADES PER ANIMAL
dfrmetadat = pd.read_excel(os.path.join(os.path.dirname(cfg.zifsh_path),'zebrafish_metadata.xlsx'))
counts = dfrmetadat.groupby(['date', 'fish_idx']).count()
indices = [''.join(t[0].split('/'))+'_'+str(t[1]) for t in counts.index]
counts = np.array(counts).flatten() * 2 # *2 because each fish has two eyes
indices_considered, counts_considered = np.unique(np.concatenate((zfish_data.id_nt, zfish_data.id_tn)), return_counts=True)
percentage_considered = np.zeros_like(counts) * np.nan
cnts_considered = np.zeros_like(counts) * np.nan
for i, id in enumerate(indices):
    try:
        percentage_considered[i] = counts_considered[indices_considered == id] / counts[i] * 100
        cnts_considered[i] = counts_considered[indices_considered == id]
    except ValueError:
        percentage_considered[i] = 0
        cnts_considered[i] = 0

dfr_zf_counts = pd.DataFrame({'fish_ID': indices, 'n_total_saccades': counts, 'n_considered_saccades': cnts_considered.astype(int),
                              'percentage_considered': percentage_considered})
dfr_zf_counts.to_csv(os.path.join(cfg.save_path, 'zf_saccade_counts.csv'), index=False)

