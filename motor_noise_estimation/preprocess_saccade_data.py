#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import function_scripts as fnc
import scipy.io as sio
import numpy as np
import config as cfg
import os

# Script to preprocess and save saccade data for zebrafish and macaque for the further downstream noise estimation analysis.

# 1) MACAQUE
print('MACAQUE')
mfn = 'example_saccades.mat' # Macaque data file name
# Load data
#Macaque saccade data
filename = os.path.join(cfg.macaque_path, mfn)
dataset = sio.loadmat(filename)
globals().update(dataset)
mytime = np.squeeze(mytime) #this is from updated globals, dont mind the error

# PREPROCESSING
# --------------
#convert saccades
dl = fnc.macaque_saccade_convert_eye_position(down_left_horizontal, down_left_vertical)
dr = fnc.macaque_saccade_convert_eye_position(down_right_horizontal, down_right_vertical)
ul = fnc.macaque_saccade_convert_eye_position(up_left_horizontal, up_left_vertical)
ur = fnc.macaque_saccade_convert_eye_position(up_right_horizontal, up_right_vertical)


# Preallocate stuff
# Onset arrays
onsetdl = np.zeros(dl.shape[0]).astype(int) #saccade onset towards down left
onsetdr = np.zeros(dr.shape[0]).astype(int) #saccade onset down right
onsetul = np.zeros(ul.shape[0]).astype(int) #saccade onset up left
onsetur = np.zeros(ur.shape[0]).astype(int) #saccade onset up right
# Offset arrays
offsetdl = np.zeros(dl.shape[0]).astype(int) #saccade offset towards down left
offsetdr = np.zeros(dr.shape[0]).astype(int) #saccade offset down right
offsetul = np.zeros(ul.shape[0]).astype(int) #saccade offset up left
offsetur = np.zeros(ur.shape[0]).astype(int) #saccade offset up right

# Saccade detection
for sac, onset, offset, direc in zip([dl, dr, ul, ur],
                      [onsetdl, onsetdr, onsetul, onsetur],
                      [offsetdl, offsetdr, offsetul, offsetur],
                      ['down left', 'down right', 'up left', 'up right']):
    print(f'Detecting saccades for {direc}')
    for i in range(sac.shape[0]):
        # print(i)
        saconidx, sacoffidx, smoothtr = fnc.detect_saccades_v2(sac[i, :], **cfg.macdet)
        onset[i] = int(saconidx)  # onset idx
        offset[i] = int(sacoffidx)  # offset idx




# 2) ZEBRAFISH
print('ZEBRAFISH')
saccadedata, datlens, nmnposidx, saccadedataout, saccadedatanoise, fish_ID, fish_ID_out, fish_ID_noise = fnc.extract_saccade_data(cfg.zifsh_path, cfg.angthres, cfg.flthres)
# Get the number of cases where eye direction is wrong and the cases where amplitude for both eyes is smaller than 5 degrees:
eyewrongidx = [i for i in range(saccadedataout.shape[2]) if (np.sign(np.mean(saccadedataout[-20:,0,i]-saccadedataout[:20,0,i])) != \
                                                             np.sign(np.mean(saccadedataout[-20:,1,i]-saccadedataout[:20,1,i])))]
lowampidx = [i for i in range(saccadedataout.shape[2]) if (np.mean(np.abs(saccadedataout[:20,0,i]-saccadedataout[-20:,0,i])) < cfg.angthres and\
                                                           np.mean(np.abs(saccadedataout[:20,1,i]-saccadedataout[-20:,1,i])) < cfg.angthres)]


tracesnt = []  # eye traces nasal to temporal
tracestn = []  # eye traces temporal to nasal
tracesntsmooth = []  # eye traces nasal to temporal
tracestnsmooth = []  # eye traces temporal to nasal
id_nt = []  # fish ID for nasal to temporal
id_tn = []  # fish ID for temporal to nasal

onsetnt = []  # saccade onset nasal to temporal
offsetnt = []  # saccade offset nasal to temporal
onsettn = []  # saccade onset temporal to nasal
offsettn = []  # saccade offset temporal to nasal

for i, id in zip(range(saccadedata.shape[2]), fish_ID):
    print(i, id)
    leraw = saccadedata[:, 0, i][~np.isnan(saccadedata[:, 0, i])]
    reraw = saccadedata[:, 1, i][~np.isnan(saccadedata[:, 1, i])]

    if i == 7:  # for this value a narrower kernel works better.
        saconidxle, sacoffidxle, lesmooth = fnc.detect_saccades_v2(leraw, smoothsigma=10, savgolorder=cfg.savgolorder)
        saconidxre, sacoffidxre, resmooth = fnc.detect_saccades_v2(reraw, smoothsigma=10, savgolorder=cfg.savgolorder)

    else:
        saconidxle, sacoffidxle, lesmooth = fnc.detect_saccades_v2(leraw, savgolorder=cfg.savgolorder)
        saconidxre, sacoffidxre, resmooth = fnc.detect_saccades_v2(reraw, savgolorder=cfg.savgolorder)

    # separate eye traces into nasal-->temporal and temporal-->nasal
    # for left eye, nasal-->temporal is positive saccades, and temporal-->nasal negative. This is opposite for right eye.
    if leraw[-1] - leraw[0] < 0:
        tracestn.append(leraw)
        tracestnsmooth.append(lesmooth)

        onsettn.append(saconidxle)
        offsettn.append(sacoffidxle)

        id_tn.append(id)

    else:
        tracesnt.append(leraw)
        tracesntsmooth.append(lesmooth)

        onsetnt.append(saconidxle)
        offsetnt.append(sacoffidxle)

        id_nt.append(id)

    if reraw[-1] - reraw[0] > 0:
        tracestn.append(reraw)
        tracestnsmooth.append(resmooth)

        onsettn.append(saconidxre)
        offsettn.append(sacoffidxre)

        id_tn.append(id)

    else:
        tracesnt.append(reraw)
        tracesntsmooth.append(resmooth)

        onsetnt.append(saconidxre)
        offsetnt.append(sacoffidxre)

        id_nt.append(id)

# MANUALLY REMOVE TRACES as they do not look like anything.
rmnt = [4, 63]
rmtn = [3, 4, 63]

# Remove the nonsaccade noise traces
tracesnt = [j for i, j in enumerate(tracesnt) if i not in rmnt]
tracesntsmooth = [j for i, j in enumerate(tracesntsmooth) if i not in rmnt]

onsetnt = [j for i, j in enumerate(onsetnt) if i not in rmnt]
offsetnt = [j for i, j in enumerate(offsetnt) if i not in rmnt]

tracestn = [j for i, j in enumerate(tracestn) if i not in rmtn]
tracestnsmooth = [j for i, j in enumerate(tracestnsmooth) if i not in rmtn]

onsettn = [j for i, j in enumerate(onsettn) if i not in rmtn]
offsettn = [j for i, j in enumerate(offsettn) if i not in rmtn]

id_nt = [j for i, j in enumerate(id_nt) if i not in rmnt]
id_tn = [j for i, j in enumerate(id_tn) if i not in rmtn]

onsetnt = np.array(onsetnt)
offsetnt = np.array(offsetnt)
onsettn = np.array(onsettn)
offsettn = np.array(offsettn)

id_nt = np.array(id_nt)
id_tn = np.array(id_tn)


# PRINT finally some descriptive stuff about number of saccades used (discarded), avg / min / max durations of the saccades used
# 1 MACAQUE
print(' MACAQUE')
print(' TOTAL NUMBER OF SACCADES: %i \n' % (len(onsetdl) + len(onsetdr) + len(onsetul) + len(onsetur)),
      'AVG duration [ms]: %.2f \n' % np.mean(np.concatenate([offsetdl-onsetdl, offsetdr-onsetdr, offsetul-onsetul, offsetur-onsetur])),
      'MIN duration [ms]: %.2f \n' % np.min(np.concatenate([offsetdl-onsetdl, offsetdr-onsetdr, offsetul-onsetul, offsetur-onsetur])),
      'MAX duration [ms]: %.2f \n' % np.max(np.concatenate([offsetdl-onsetdl, offsetdr-onsetdr, offsetul-onsetul, offsetur-onsetur])))
# 2 ZEBRAFISH
print(' ZEBRAFISH')
print(' TOTAL NUMBER OF SACCADES: %i \n' % (len(onsetnt) + len(onsettn)),
      'AVG duration [ms]: %.2f \n' % np.mean(np.concatenate([offsetnt-onsetnt, offsettn-onsettn])),
      'MIN duration [ms]: %.2f \n' % np.min(np.concatenate([offsetnt-onsetnt, offsettn-onsettn])),
      'MAX duration [ms]: %.2f \n' % np.max(np.concatenate([offsetnt-onsetnt, offsettn-onsettn])))



# SAVE EVERYTHING FOR USING DOWNSTREAM
# 1) Save macaque data
print('Saving macaque saccade data...')
np.savez(os.path.join(cfg.save_path, 'macaque_saccade_data_processed.npz'),
         dl=dl, dr=dr, ul=ul, ur=ur,
         onsetdl=onsetdl, onsetdr=onsetdr, onsetul=onsetul, onsetur=onsetur,
         offsetdl=offsetdl, offsetdr=offsetdr, offsetul=offsetul, offsetur=offsetur,
         mytime=mytime)
# 2) Save zebrafish data
print('Saving zebrafish saccade data...')
np.savez(os.path.join(cfg.save_path, 'zebrafish_saccade_data_processed.npz'),
         tracesnt=np.array(tracesnt, dtype=object), tracestn=np.array(tracestn, dtype=object),
         tracesntsmooth=np.array(tracesntsmooth, dtype=object), tracestnsmooth=np.array(tracestnsmooth, dtype=object),
         onsetnt=onsetnt, offsetnt=offsetnt,
         onsettn=onsettn, offsettn=offsettn,
         id_nt=id_nt, id_tn=id_tn,
         saccadedataout=np.array(saccadedataout, dtype=object), saccadedatanoise=np.array(saccadedatanoise, dtype=object),
         allow_pickle=True)
