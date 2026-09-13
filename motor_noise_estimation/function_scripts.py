# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:25:43 2020

@author: Ibrahim Alperen Tunc
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from skimage import io, filters
from scipy.stats import reciprocal #log normal distribution
import random
from scipy.ndimage.filters import correlate1d
from scipy.signal import savgol_filter
from PIL import Image
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
from IPython import embed
import config as cfg

#Script for functions used in the project


def extract_saccade_data(root, angthres, flthres, mfd=os.path.join(os.path.dirname(cfg.zifsh_path), 'zebrafish_metadata.xlsx')):
    """
    Extract the eye position saccade data from the .txt files
    
    Parameters
    -----------
    root: string
        The string of datapath which contains all data files.
    angthres: float
        The threshold for the saccade amplitude in degrees. Data is discarded if both eyes have a saccade magnitude smaller than threshold.  
    flthres: float
        The threshold to discard data based on noise fluctuations. If the difference between 2 datapoints is bigger than this value times maximum
        value difference, then the data is discarded. Note that this value is between 0 and 1.
    mfd: string, optional
        The string of the metadata file path. This is used to extract the fish index and date of the data files.
        Default is '../data/zebrafish_metadata.xlsx'
    Returns
    -------
    saccadedata: 3-D array (npos x 2 x ntrial)
        The data array containing the eye positions. First dimension is the eye position angular value over time, second dimension is the eye 
        position (0 is left eye, 1 is righr eye) and the last dimension is for trial index
    datlens: 1-D array
        The time length of each trial. Note that the length is not corrected by the sampling rate
    nmnposidx: 1-D array
        The index array of trials which deviate with their time length 
    saccadedataout: 3-D array (npos x 2 x ntrialout)
        The discarded data array containing the eye positions. Dimensions same as in saccadedata. This is kept just to check if taken out data is 
        indeed non-saccade
    saccadedatanoise: 3-D array (npos x 2 x ntrialout)
        The noisy (flthres determined) data array containing the eye positions. Dimensions same as in saccadedata. This is kept just to check if 
        noise threshold works well
    fish_ID: 1-D array
        The fish ID over the entire recording for the considered saccades. This will be a string of DATE_fishidx.
    fish_ID_out: 1-D array
        The fish ID over the entire recording for removed saccades. This will be a string of DATE_fishidx.
    fish_ID_noise: 1-D array
        The fish ID over the entire recording for noisy saccades. This will be a string of DATE_fishidx.
    """
    #Import the saccade data txt files and combine them together for further analysis
    
    #Insert the external harddrive data path
    sys.path.insert(0, root)
    
    saccadefiles = [] #the list of saccade data (eye position) txt files    
    saccadedatal = [] #extract the data from the data files to the list (hence 'l' in the end)
    saccadefilesout = [] #the list of discarded saccade data (eye position) txt files
    saccadedatalout = [] #extract the data from the data files to the list (hence 'l' in the end, discarded data)
    saccadefilesnoise = [] #the list of noisy saccade data (eye position) txt files
    saccadedatalnoise = [] #extract the data from the data files to the list (hence 'l' in the end, discarded data)
    fish_ID = [] #the list of fish ID for the considered saccades
    fish_ID_out = [] #the list of fish ID for the discarded saccades
    fish_ID_noise = [] #the list of fish ID for the noisy saccades

    metadat = pd.read_excel(mfd) #read the metadata file to extract the fish index and date of the data files

    #extract the eye position txt files
    for path, subdirs, files in os.walk(root):
        for name in files:
            filename, file_extension = os.path.splitext(name)
            #print(filename, file_extension)
            if file_extension == '.txt' and filename[0:6] == 'eyepos':
                saccadefiles.append(os.path.join(path, name))
        
    for filename in saccadefiles:
        data = pd.read_csv(filename, sep='\t', header=None)
        # Get the fish index
        date = metadat.loc[metadat['file'] == os.path.basename(filename), 'date'].values[0]
        idx = int(metadat.loc[metadat['file'] == os.path.basename(filename), 'fish_idx'].values[0])
        fishID = '_'.join([''.join(date.split('/')), str(idx)])

        #find local maxima of the data showing fluctuations bigger than flthres. Discard the first and last few seconds as some recordings show
        #not problematic fluctuations at the onset
        peaksleft, proml = find_peaks(np.array(data[4])[15:-15], prominence=flthres*(np.max(np.array(data[4]))-np.min(np.array(data[4]))))
        peaksright, promr = find_peaks(np.array(data[5])[15:-15], prominence=flthres*(np.max(np.array(data[5]))-np.min(np.array(data[5]))))

        
        #discard the data if the sign of the average difference between angle position of last 20 ms and first 20 ms does not match for both eyes 
        #(first 2 lines) or if this average difference is smaller than angthreshold for BOTH eyes (saccade too small, last 2 lines)
        if (np.sign(np.mean(np.array(data[4])[-20:]-np.array(data[4])[:20])) != \
            np.sign(np.mean(np.array(data[5])[-20:]-np.array(data[5])[:20]))) or \
            np.mean(np.abs(np.array(data[4])[:20]-np.array(data[4])[-20:])) < angthres and\
            np.mean(np.abs(np.array(data[5])[:20]-np.array(data[5])[-20:])) < angthres:
                        
            saccadefilesout.append(os.path.join(path, name))
            saccadedatalout.append((np.array(data[4]),np.array(data[5]))) #order is LE RE
            fish_ID_out.append(fishID)

        #add the data to noisy list if any 2 timepoint shows fluctuations more than flthres * (max(data) - min(data)). This for any of the eye
        elif len(peaksleft)!=0 or len(peaksright)!=0:
           saccadefilesnoise.append(os.path.join(path, name))
           saccadedatalnoise.append((np.array(data[4]),np.array(data[5]))) #order is LE RE
           fish_ID_noise.append(fishID)
        
        #add the data as saccade if none of the above holds.
        else:        
            saccadedatal.append((np.array(data[4]),np.array(data[5]))) #order is LE RE
            fish_ID.append(fishID)

    #!npos (number of datapoints for each saccade trial) is not equal for all trials
    datlens = [] #length of the trials
    datlensout = [] #length of the discarded trials
    datlensnoise = [] #length of the noisy trials

    
    #save lengths of all trials.
    for dat in saccadedatal:
        datlens.append(len(dat[0]))
    
    for dat in saccadedatalout:
        datlensout.append(len(dat[0]))
    
    for dat in saccadedatalnoise:
        datlensnoise.append(len(dat[0]))
    
    nmnposidx = np.squeeze(np.where(datlens!=np.median(datlens))) #the indices of the trials which have different length than most of the trials
    
    #adjust the data structure accordingly : datamatrix = npos x 2 x ntrials, 2 is for LE RE
    #datamatrix is adjusted in such a way, that the first dimension has the length of the maximum trial.        
    saccadedata = np.empty((np.max(datlens) ,len(saccadedatal[0]), len(saccadedatal)))
    saccadedata[:] = np.nan
    saccadedataout = np.empty((np.max(datlensout) ,len(saccadedatalout[0]), len(saccadedatalout)))
    saccadedataout[:] = np.nan
    saccadedatanoise = np.empty((np.max(datlensnoise) ,len(saccadedatalnoise[0]), len(saccadedatalnoise)))
    saccadedatanoise[:] = np.nan    
    
    #fill in the values for saccadedata. This will be the same loop as above (loop over saccadedatal), but for general usability of the code, I guess
    #it is better to first estimate the maximum length of the trial and then adjust accordingly the data matrix.
    for idx, dat in enumerate(saccadedatal):
        saccadedata[:datlens[idx],:,idx] = np.array(dat).T
        assert saccadedata.shape[-1] == len(fish_ID), "The number of trials in saccadedata does not match the number of fish IDs."
    #same as above but for discarded data
    for idx, dat in enumerate(saccadedatalout):
        saccadedataout[:datlensout[idx],:,idx] = np.array(dat).T
        assert saccadedataout.shape[-1] == len(fish_ID_out), "The number of trials in saccadedataout does not match the number of fish IDs."

    #same as above but for noisy data
    for idx, dat in enumerate(saccadedatalnoise):
        saccadedatanoise[:datlensnoise[idx],:,idx] = np.array(dat).T
        assert saccadedatanoise.shape[-1] == len(fish_ID_noise), "The number of trials in saccadedatanoise does not match the number of fish IDs."


    return saccadedata, datlens, nmnposidx, saccadedataout, saccadedatanoise, fish_ID, fish_ID_out, fish_ID_noise


def detect_saccades_v2(rawdata, smoothsigma=15, velthres=1000, accthres=100000, savgollength=51, savgolorder=4, 
                       rf=1000, a=0.05, b=0.05, velperc=90, onstd=3, macaque=False):
    """
    Improved saccade onset and offset detection algorithm adapted from Nyström & Holmqvist 2010.
    
    Parameters
    ----------
    rawdata: 1-D array
        The raw saccade trace data
    rf: float, optional
        Temporal sampling frequency in Hz
    smoothsigma: float, optional
        The standard deviation of the Gaussian smoothing kernel
    savgollength: float, optional
        The length of the Savitzky-Golay filter in ms
    savgolorder: float, optional
        The polynomial order of the Savitzky-Golay fit. See also scipy.signal.savgol_filter documentation
    velthres: float, optional
        The velocity threshold for the filtered data in °/s. Any velocity bigger than this value is physiologically unrealistic
        and is therefore discarded.
    accthres: float, optional
        The acceleration threshold for the filtered data in °/s^2. Any acceleration bigger than this value is physiologically 
        unrealistic and is therefore discarded.
    a: float, optional
        Weight factor of the global noise for saccade offset detection
    b: float, optional
        Weight factor of the local noise for saccade offset detection
    saconidx: integer, optional
        In one instance, saccade onset cannot be reliably detected by the algorithm, since the onset is at 0. Thus, this
        optional variable is used to specify the saccade onset index manually when needed.
    velperc: float, optional
        Percentile for the saccade velocity threshold. Can be between 0 and 100.
    onstd: float
        Standard deviation value used during saccade onset estimation.
    macaque: boolean, optional
        If True, no Gaussian smoothing is done since it is not necessary for macaque.
    
    Returns
    -------
    saconidx: integer
        The index of the saccade onset
    sacoffidx: integer
        The index of the saccade offset
    """
    if macaque == True:
        smoothdata = rawdata #no data smoothing with Gaussian if macaque saccades are used 
    
    else:
        radius = int(4 * smoothsigma**2 + 0.5)
    
        x = np.arange(-radius, radius+1)
        gausskern = np.exp(-0.5 / smoothsigma**2 * x ** 2)
        gausskern[x<-10] = 0 #causal kernel, so for x<0 all values are zero
        gausskern /= np.sum(gausskern)
        smoothdata = correlate1d(rawdata, gausskern[::-1])
        
    velocitydata = np.abs(savgol_filter(smoothdata, savgollength, savgolorder, deriv=1) * rf) #filtered LE velocity in °/s
    #if the last velocity point in the data is extremely high, this likely indicates a recording artifact, so velocity is set to 0
    if velocitydata[-1] > 20: 
        velocitydata[-50:] = 0
    #discard velocity values bigger than threshold
    velocitydata[velocitydata>velthres] = None
    accdata = np.abs(savgol_filter(smoothdata, savgollength, savgolorder, deriv=2) * rf**2) #filtered LE acceleration in °/s^2
    #discard data points exceeding acceleration threshold.
    velocitydata[accdata>accthres] = None
    
    #saccade velocity threshold estimation: iterative and data-driven approach
    sacvelthres = np.percentile(velocitydata[~np.isnan(velocitydata)],velperc) #set the initial threshold to 99th percentile to be on the safe side
    #iteration
    thres = False
    iternum = 0
    stdval = 6
    while thres == False:
        iternum += 1
        underthres = velocitydata[velocitydata<sacvelthres]
        previoussacvelthres = sacvelthres
        sacvelthres = np.mean(underthres[~np.isnan(underthres)]) + stdval*np.std(underthres[~np.isnan(underthres)])
        if sacvelthres >= np.max(velocitydata):
            stdval -= 0.5
        elif np.abs(previoussacvelthres-sacvelthres) < 1:
            #print(iternum)
            thres = True
    
    #saccade onset detection: saccade onset velocity threshold is defined as mean+3*std for eye traces lower than peak threshold
    uthresidx = np.where(velocitydata>=sacvelthres)[0][0] #index of first peak exceeding velocity threshold
    if uthresidx == 0:
        uthresidxs = np.where(velocitydata>=sacvelthres)[0] #indices of the datapoints over threshold
        print('first index skipped')
        #first datapoints likely to have some fluctuations, thus take the index which shows a difference bigger than 1 
        #compared to previous index.
        uthresidx = uthresidxs[np.where(np.diff(uthresidxs)>1)[0][0]+1] 
    underthres = velocitydata[0 : uthresidx]
    saconthres = np.mean(underthres) + onstd*np.std(underthres)
    saconidx = np.where((underthres[1:] < saconthres) & (np.diff(underthres)>=0))[0][-1] 
    
    #Saccade offset detection: choose the leftmost velocity peak and search forward from there to find wished saccade.
    offpeakidx = np.where(velocitydata>=sacvelthres)[0][-1]
    underthres = velocitydata[offpeakidx:] #saccade velocity trace from last velocity peak on
    
    if saconidx < 40:
        presaccade = velocitydata[:saconidx] #velocity curve before saccade onset 
    else:
        presaccade = velocitydata[saconidx-40:saconidx] #velocity curve before saccade onset
    
    noisefacs = False
    while noisefacs == False:
        noisefac = np.mean(presaccade) + 3*np.std(presaccade) #adaptive noise factor
        sacoffthres = a*saconthres + b*noisefac
        
        while sacoffthres < np.min(underthres):
            a += 0.005
            b += 0.005
            sacoffthres = a*saconthres + b*noisefac
        sacoffidx = np.where((underthres[1:] < sacoffthres) & (np.diff(underthres)<=0))[0][0] + offpeakidx
        
        glitw = 400 #glissade time window
        if sacoffidx + glitw > len(velocitydata):
            glitw = len(velocitydata) - glitw #if time window bigger than total eye trace length, it is set to the end point of the trace
        else:
            pass
        
        if True in (velocitydata[sacoffidx:sacoffidx+glitw] >= sacoffthres):
           print("glissade detected")
           #Find the glissade offset
           glipeakidx = np.where(velocitydata[sacoffidx:sacoffidx+glitw]>=sacoffthres)[0][-1] + sacoffidx
           gliunderthres = velocitydata[glipeakidx:]
           if len(np.where((gliunderthres[1:] < sacoffthres) & (np.diff(gliunderthres)<=0))[0]) > 0:
               noisefacs = True
            
           else:
               a += 0.005
               b += 0.005
               continue
               
           if len(gliunderthres) == 1:
               sacoffidx = glipeakidx
           else:
               sacoffidx = np.where((gliunderthres[1:] < sacoffthres) & (np.diff(gliunderthres)<=0))[0][0] + glipeakidx    
        else:
            noisefacs = True
            
    return saconidx, sacoffidx, smoothdata


#Fit function from Dai et al. 2016
def saccade_fit_func(t, c, nu, tau, t_0, s_0):
    """
    Parametric fit function for saccade trace from Dai et al. 2016
    
    Parameters
    ----------
    t: 1-D array
        The time array in ms
    c: float
        Model parameter
    nu: float
        Model parameter
    tau: float
        Model parameter
    t_0: float
        Model parameter
    s_0: float
        Model parameter
        
    Returns
    -------
    fitfunc: 1-D array
        The fitted function to the saccade trace    
    """
    comp1 = c*saccade_fit_func_f(nu*(t-t_0)/c)
    comp2 = -c*saccade_fit_func_f(nu*((t-t_0)-tau)/c)
    fitfunc = comp1 + comp2 + s_0    
    #if return_comps == True:
    #    return fitfunc, comp1, comp2
    #else:
        #return fitfunc
    return fitfunc


def saccade_fit_func_f(t):
    """
    Part of the fit function to be used in the saccade fitting.
    
    Parameters
    ----------
    t: 1-D array
        The time array
    
    Returns
    -------
    f: 1-D array
        Ramp function used for saccade fitting
    """
    f = np.zeros(len(t))
    f[t<=0] = 0.25*np.e**(2*t[t<=0])
    f[t>=0] = t[t>=0] + 0.25*np.e**(-2*t[t>=0])
    return f


def linear_fit(x, a, b):
    """
    Function used for linear fit.
    
    Parameters
    ----------
    x: 1-D array
        The data used for the linear fit
    a: float
        Slope of the fit curve
    b: float
        Offset of the fit curve
    """
    return a*x+b


def macaque_xi_estimation(sacarr, onset, offset):
    """
    Estimate the additive motor noise xi.

    Parameters
    ----------
    sacarr : 2-D array
        Saccade array. Shape ntrials * time
    onset : 1-D array
        Array of onset indices for each trial.
    offset : 1-D array
        Array of offset indices for each trial.

    Returns
    -------
    xi : 1-D array
        Motor error for each time trace in all saccade trials.
    prenum : int
        Number of used presaccadic traces.
    postnum : int
        Number of used postsaccadic traces.
    xisvals : tuple
        The standard deviation parameters calculated in different methods. See function calculate_s_xi.        
    """
    xi = [] 
    prenum = 0 #number of presaccadic traces used in the analysis
    postnum = 0 #number of postsaccadic traces used in the analysis
    for idx in range(sacarr.shape[0]):
        presaccade = sacarr[idx, :onset[idx]]#discard 5 ms from beginning and end of presaccadic fixation
        postsaccade = sacarr[idx, offset[idx]:] #dicard again
    
        if len(presaccade) < 20: #if less than 20 ms available for presaccade, discard the presaccadic curve from analysis
            None        
        else:
            prenum += 1
            presaccade = sacarr[idx, 5:onset[idx]-5]#discard 5 ms from beginning and end of presaccadic fixation
            xpre = np.arange(0,len(presaccade))
            preparams, _ = curve_fit(linear_fit, xpre, presaccade)
            prefit = linear_fit(xpre, *preparams)
            xi.append(presaccade-prefit)
            #print(presaccade-prefit)
        if len(postsaccade) < 20: #same story
            None
        else:
            postnum += 1
            postsaccade = sacarr[idx, offset[idx]+5:-5] #discard again
            xpost = np.arange(0,len(postsaccade))
            postparams, _ = curve_fit(linear_fit, xpost, postsaccade)
            postfit = linear_fit(xpost, *postparams)
            xi.append(postsaccade-postfit)
    xi = np.array([a for b in xi for a in b])
    xisvals = calculate_s_xi(xi)
    return xi, prenum, postnum, xisvals


def calculate_s_xi(xi):
    """
    Calculate s_xi using different data exclusion criteria.

    Parameters
    ----------
    xi : 1-D array
        The additive noise array.

    Returns
    -------
    s_xiraw : float
        Standard deviation of xi without removing any outliers.
    s_xifiltered : float
        Standard deviation of xi with removing outliers. Outliers defined as data points outside 1.5 IQR.
    s_xipercentiled : float
        Standard deviation of xi with removing outliers. Outliers defined as data points above 99th percentile.
    """
    #find outliers
    xiiqr = np.percentile(xi,75)-np.percentile(xi,25)
    xifiltered = xi[(xi>np.median(xi)-1.5*xiiqr) & (xi<np.median(xi)+1.5*xiiqr)]
    s_xiraw = np.std(xi) 
    #outliers as 1.5 IQR
    s_xifiltered = np.std(xifiltered)
    #outliers as above 99th percentile (probably this is the safest way to go)
    s_xipercentiled = np.std(xi[(xi<np.percentile(xi, 99.5)) & (xi>np.percentile(xi, 0.5))]) 
    return s_xiraw, s_xifiltered, s_xipercentiled


def macaque_eps_estimation(sacarr, xi, onset, offset, RMSfac=cfg.RMSfac, ucutoff=cfg.ucut, pre_post_dur=50, return_RMS=False, return_fitparams=False):
    """
    Estimate multiplicative motor noise epsilon (saccadic, during the presence of motor command)

    Parameters
    ----------
    sacarr : 2-D array
        Saccade array. Shape ntrials * time
    xi : 1-D array
        The additive noise array.
    onset : 1-D array
        Array of onset indices for each trial.
    offset : 1-D array
        Array of offset indices for each trial.
    RMSfac : float, optional
        The RMS threshold factor for the saccade fitting. Saccades with RMS values lower than mean-RMSfac*std are considered.
        The default is taken from config.py. This was not used for the publication and was therefore set to -9999.
    ucutoff : float, optional
        Cutoff value for the command input. The default is taken from the config script.
    onsrem : int, optional
        The index until which the saccade onset will be discarded. The default is 5.
    offsrem : int, optional
        The index until from which on the saccade offset will be discarded. The default is 150.
    fplot : boolean, optional
        If True, fit and noise parameter plots are generated for each saccade. The default is False.
    nplot : boolean, optional
        If True, nice saccades with their respective fit functions are shown. The default is False.
    dt : float, optional
        Time step in seconds. The default is 0.001 i.e. sampling rate of 1000 Hz. Both macaque and zf data should accomodate for this.
    return_RMS : boolean, optional
        If True, the RMS values of the saccade fitting are returned. The default is False.
    return_fitparams : boolean, optional
        If True, the fitted saccade traces are returned. The default is False.
    Returns
    -------
    ul : 1-D array
        Command input array.
    epsl : 1-D array
        Multiplicative noise array. This array is synched with ul
    s_eps : float
        Estimated standard deviation of epsilon.

    """
    
    #Motor noise 2: epsilon -> trace during saccade
    #u is temporal derivative of the saccadic trace template, epsilon is calculated as s_eps = sqrt(var_tot-var_xi)/u.
    #Template from Dai et al. 2016
    #choose nice saccades -> ones fitting the template well, use RMS to quantify this.
    
    #preallocation
    nl = []
    ul = []
    RMSs = np.zeros(sacarr.shape[0])
    fittedsacs = []
    fitpars = []
    varxi = np.var(xi)

    #do the fitting
    for idx in range(sacarr.shape[0]):
        # if offset[idx]-offsrem <= onset[idx]+onsrem or onset[idx]+onsrem >= offset[idx]-offsrem:
        #     print('Saccade too short for removing %.0f ms from beginning and %.0f from end. This saccade is skipped' \
        #               %(onsrem,offsrem))
        #     continue
        preidx = onset[idx]-pre_post_dur
        postidx = offset[idx]+pre_post_dur
        if preidx < 0:
            preidx = 0
        if postidx > sacarr.shape[1]:
            postidx = sacarr.shape[1]
        saccade = sacarr[idx, preidx:postidx] #saccade trace
        
        if len(saccade) == 0:
            print('Saccade length is zero!')
            continue
        t = np.arange(0, len(saccade))
        fitparams, _ = curve_fit(saccade_fit_func, t, saccade, method='lm', maxfev=100000)
        fittedsac = saccade_fit_func(t, *fitparams)
        totnoise = (saccade-fittedsac)
        totnoise = totnoise[1:] # AS OF AUGUST 2026 THIS PART IS DEBUGGED NOW THE EPSILONS SHOULD BE ALIGNED!
        utemp = np.abs(np.diff(fittedsac)) #template u #TODO: Possibly ADD HERE THE TEMPORAL DERIVATIVE PART I.E. DIVIDE BY DT!!! then this is in seconds.
        #eps = np.sqrt(totnoise-varxi)[1:] / utemp
        nl.append(totnoise)
        ul.append(utemp)
        RMSs[idx] = np.sqrt(np.mean(totnoise**2))
        fittedsacs.append(fittedsac)
        fitpars.append(fitparams)
        # print(idx)

    rmsidx = ~np.isnan(RMSs)
    nicesacidxs = np.where(RMSs <= np.mean(RMSs[rmsidx]) - RMSfac*np.std(RMSs[rmsidx]))[0]

    # nl = np.array(nl)
    # nl = np.array([a for b in nl[nicesacidxs] for a in b])
    # ul = np.array(ul)
    # ul = np.array([a for b in ul[nicesacidxs] for a in b])
    nl = np.concatenate([nl[i] for i in nicesacidxs])
    ul = np.concatenate([ul[i] for i in nicesacidxs])
    assert len(nl) == len(ul), "Noise and command input arrays are not the same length!"
    (ul, nl) = zip( *sorted( zip(ul, nl) ) ) #sort u and noise arrays in ascending u order TODO BUG!
    # ul = np.array(ul)
    # nl = np.array(nl)

    #bin u values and epsilon values -> use the percentile approach you also used in zf
    nperbin = 100 #number of datapoints per bin

    ubins = []
    binnedu = [] #u bins
    binnedn = [] #total noise bins
    
    for k in range(np.ceil(len(ul)/nperbin).astype(int)):
        # print(k*nperbin, k*nperbin+nperbin)
        if k*nperbin+nperbin < len(ul):
            us = ul[k*nperbin : k*nperbin+nperbin]
            ns = nl[k*nperbin : k*nperbin+nperbin]
        else:
            us = ul[k*nperbin:]
            ns = nl[k*nperbin:]
        ubins.append(np.mean(us))
        binnedu.append(us)
        binnedn.append(ns)
        
    ubins = np.array(ubins)  
    
    #find average s_eps per bin and then average over all
    usidx = np.where(ubins>ucutoff)[0][0] #the first u bin index value bigger than cutoff
    
    stdnperbin = np.array([np.std(a) for a in binnedn]) #standard deviation of total noise
    s_epss = 1/np.abs(ubins[usidx:])*np.sqrt(stdnperbin[usidx:]**2 - varxi)
    s_eps = np.mean(s_epss[~np.isnan(s_epss)]) 

    if not return_RMS and not return_fitparams:
        return s_eps, ubins, s_epss, nl, stdnperbin, nicesacidxs

    rest_return = {} # dictionary to store the results to be returned rest.

    if return_RMS:
        rest_return['RMS'] = RMSs
    if return_fitparams:
        rest_return['fitparams'] = fitpars
    return s_eps, ubins, s_epss, nl, stdnperbin, nicesacidxs, rest_return


class coordinate_transformations():
    """Class for functions to convert geographical coordinates to cartesian and vice versa."""

    def car2geo(x, y, z):
        """
        Convert cartesian coordinate system to geographical.

        Parameters
        ----------
        x : float/ 1-D array
            x value in cartesian coordinates.
        y : float/ 1-D array
            y in cartesian coordinates.
        z : float/ 1-D array
            z in cartesian coordinates.

        Returns
        -------
        r: float/ 1-D array
            radius (distance from origin)
        azimuth: float/ 1-D array
            azimuth in deg (horizontal angle)
        elevation: float/ 1-D array
            elevation in deg (vertical angle)
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        elevation = np.arcsin(z / r)
        azimuth = np.arctan2(y, x)

        return r, np.rad2deg(azimuth), np.rad2deg(elevation)

    def geo2car(r, azimuth, elevation):
        """
        Convert geographical coordinate system to cartesian.

        Parameters
        ----------
        r: float/ 1-D array
            radius (distance from origin)
        azimuth: float/ 1-D array
            azimuth in deg (horizontal angle)
        elevation: float/ 1-D array
            elevation in deg (vertical angle)

        Returns
        -------
        x : float/ 1-D array
            x value in cartesian coordinates.
        y : float/ 1-D array
            y in cartesian coordinates.
        z : float/ 1-D array
            z in cartesian coordinates.
        """
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)

        return x, y, z


def macaque_saccade_convert_eye_position(horizontal, vertical):
    """
    Convert horizontal and vertical eye angle positions to a single angle relative to visual field center
    (0° azimuth and elevation). The function returns the angle value
    between the vectors |viewer position -> visual field center| and |viewer position -> given eye position|.

    Parameters
    ----------
    horizontal : 2-D array
        Horizontal eye position in degrees. Shape ntrial * time
    vertical : 2-D array
        Vertical eye position in degrees.

    Returns
    -------
    dang : 2-D array
        Eye position angle relative to origin.

    """

    """
    First, convert horizontal and vertical angles to a single angle relative to visual field center (0,0)
    since the eye positions are in angles, you can use spherical visual field, convert it to cartesian, find the 
    respective distances required and finally re-convert all values to a single angle per time point per trial.
    """
    # set horizontal and vertical start positions to 0 (averaged over first 10 ms)
    horizontal -= np.tile(np.mean(horizontal[:, :10], axis=1), (horizontal.shape[1], 1)).T
    vertical -= np.tile(np.mean(vertical[:, :10], axis=1), (vertical.shape[1], 1)).T

    # convert eye position to cartesian coordinates
    x, y, z = coordinate_transformations.geo2car(np.ones(horizontal.shape), horizontal, vertical)
    # origin in cartesian coordinates
    xor, yor, zor = coordinate_transformations.geo2car(1, 0, 0)
    # distance between visual field origin and datapoints:
    dist = np.sqrt((x - xor) ** 2 + (y - yor) ** 2 + (z - zor) ** 2)
    # convert distance to angle, #while keeping the angle signs in mind
    dang = 2 * np.rad2deg(np.arcsin(dist / 2))  # * np.sign((x-xor)+(y-yor)+(z-zor))
    # set presaccadic eye position to 0 (averaged over 10 ms)
    dang -= np.tile(np.mean(dang[:, :10], axis=1), (dang.shape[1], 1)).T
    return dang
