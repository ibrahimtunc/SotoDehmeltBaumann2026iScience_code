% This code was prepared by Florian Alexander Dehmelt at Tuebingen 
% University and published in 2026 under Creative Commons license 
% CC BY-SA-NC 4.0 to accompany the following scientific publication: 
% 
% Giulia Soto*, Florian A. Dehmelt*, Matthias P. Baumann*, Ibrahim Tunç, 
% Yue Yu, Tatiana Malevich, Ziad M. Hafed**, Aristides B. Arrenberg** 
% (2026) Species-specific sensorimotor noise levels explain saccadic 
% suppression strength differences between macaques and zebrafish. iScience
% 
% *equal contributions, **corresponding authors 
%
% This code and the functions it calls incorporate prior work done by 
% Frederic Crevecoeur in conjunction with their previous publication,
% 
% Frédéric Crevecoeur, Konrad P Kording (2017) Saccadic suppression as a 
% perceptual consequence of efficient sensorimotor estimation. 
% eLife 6:e25073, https://doi.org/10.7554/eLife.25073 
% 
% We thank F.C. for sharing his code with us, and gratefully acknowledge 
% his support in reproducing the findings or their earlier paper.


clear all
close all

% optionally, fix random seed to reproduce figure between simulations
rng(111)

% choose the range of noise parameters to be simulated
motornoiselist = [1 2.2 4.7 10 22];
sensorynoiselist = [1 1.8 3.2 5.6 10];

% housekeeping to make sure variables have expected size and type
sensorynoiselist = fliplr(sensorynoiselist);
jointnoiselist = []; 
hd = struct;

for motorindex = 1:numel(motornoiselist)
  for sensorindex = 1:numel(sensorynoiselist)
    
    motornoisefactor = motornoiselist(motorindex);
    sensorynoisefactor = sensorynoiselist(sensorindex);

    % define other simulation parameters, run simulations, plot results
    hd = NoiseComparison_SimulateNoise(hd,motornoisefactor,sensorynoisefactor);
    jointnoiselist = [jointnoiselist; motornoisefactor sensorynoisefactor]; %#ok<AGROW>
    
  end
end


% arrange figure panels and add decorations
NoiseComparison_MakeFigures