function [hd] = NoiseComparison_SimulateNoise(hd,motornoisefactor,sensorynoisefactor)

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
  % This code and the function it calls incorporate prior work done by 
  % Frederic Crevecoeur in conjunction with their previous publication,
  % 
  % Frédéric Crevecoeur, Konrad P Kording (2017) Saccadic suppression as a 
  % perceptual consequence of efficient sensorimotor estimation. 
  % eLife 6:e25073, https://doi.org/10.7554/eLife.25073 
  % 
  % We thank F.C. for sharing his code with us, and gratefully acknowledge 
  % his support in reproducing the findings or their earlier paper.


  % Simulation parameters are set here:

  delta = .0015;                           % discretisation step
  Sm = motornoisefactor * 10^-2;           % motor noise scaling
  Sf = sensorynoisefactor * 10^-6;         % sensory noise scaling
  ce = 0;                                  % internal noise
  alphaNoisec = .006;                      % signal dependent noise scaling
  alphaNoised = sqrt(alphaNoisec^2/delta);
  
  simdata.delta = delta;
  simdata.fixation = 0.5;                  % initial fixation
  simdata.timefree = 0.05;                 % movement time
  simdata.timestab = .25;                  % stabilization
  
  simdata.delay = 0.1;                     % sensorimotor delay
  simdata.alpha = .01;                     % tolerance in the cost function
  simdata.noise = [ce Sf Sm];              % ce Sensory Motor
  simdata.alphad = alphaNoised;
  simdata.alphac = alphaNoisec;

  init = -10;
  amp1 = 20;
  ic = init*pi/180*[1 0]';                 % initial condition
  fc = (init+amp1)*pi/180*[1 0]';          % final condition


  % The actual simulations are run here:

  [x,control,~,~,K,~] = NoiseComparison_ControlModel(ic,fc,simdata,1);


  % Preliminary plots are generated below:


  % configure time axis
  time = 0:delta:delta*(size(x.xn,2)-1);
  b1 = 1;
  b2 = numel(time)-1;
  
  % set up preliminary figure
  numpanel = 4;
  if isfield(hd,'crevecoeur')
    if isfield(hd.crevecoeur,'ax')
      trialindex = numel(hd.crevecoeur.ax)/numpanel + 1;
    else
      trialindex = 1;
    end
  else
    hd.crevecoeur.fg = figure();
    trialindex = 1;
  end
    
  % set panel dimensions
  fontsize = 11;
  panel.width = 62;
  panel.height = 100;
  panel.hanchor = 95;
  panel.vanchor = 55;
  panel.hsep = 10;
  panel.vsep = 25;
  
  hd.crevecoeur.fg.Color = [1 1 1];
  hd.crevecoeur.fg.Position = [3 400 1916 580];
    
  hd.crevecoeur.ax(1,trialindex) = axes();
  hd.crevecoeur.ax(1,trialindex).FontSize = fontsize;
  hd.crevecoeur.ax(1,trialindex).Units = 'pixels';
  hd.crevecoeur.ax(1,trialindex).Position = [panel.hanchor panel.vanchor panel.width panel.height] ...
    + [(trialindex-1)*(panel.width+panel.hsep) 0 0 0];
  hold on
  plot(time(b1:b2),mean(x.xn(:,b1:b2),1)*180/pi,'k','LineWidth',1), hold on
  plot(time(b1:b2),mean(x.tn(:,b1:b2),1)*180/pi,'LineWidth',1,'Color',.4*[1 1 1]), hold on
  plot(time(b1:b2),mean(x.ten(:,b1:b2),1)*180/pi,':','Color',.6*[1 1 1]), hold on
  hold off
  xlabel('time (a.u.)')
  ylabel('position, target')

  hd.crevecoeur.ax(2,trialindex) = axes();
  hd.crevecoeur.ax(2,trialindex).FontSize = fontsize;
  hd.crevecoeur.ax(2,trialindex).Units = 'pixels';
  hd.crevecoeur.ax(2,trialindex).Position = ...
    hd.crevecoeur.ax(1,trialindex).Position + 1*[0 panel.height+panel.vsep 0 0];
  plot(time(b1:b2),mean(control.cn(:,b1:b2),1),'Color',[1 .4 .2],'LineWidth',1)
  xlabel('time (a.u.)')
  ylabel('control','Fontsize',fontsize)

  hd.crevecoeur.ax(3,trialindex) = axes();
  hd.crevecoeur.ax(3,trialindex).FontSize = fontsize;
  hd.crevecoeur.ax(3,trialindex).Units = 'pixels';
  hd.crevecoeur.ax(3,trialindex).Position = ...
    hd.crevecoeur.ax(1,trialindex).Position + 2*[0 panel.height+panel.vsep 0 0];
  hold on
  plot(time(b1:b2),mean(K.index(1,b1:b2),1),'LineWidth',1)
  xlabel('time')
  ylabel('FB weight')
  
  hd.crevecoeur.ax(4,trialindex) = axes();
  hd.crevecoeur.ax(4,trialindex).FontSize = fontsize;
  hd.crevecoeur.ax(4,trialindex).Units = 'pixels';
  hd.crevecoeur.ax(4,trialindex).Position = ...
    hd.crevecoeur.ax(1,trialindex).Position + 3*[0 panel.height+panel.vsep 0 0];
  hold on
  plot(time(b1:b2),mean(K.index(1,b1:b2),1) / mean(K.index(1,b2),1),'LineWidth',1)
  xlabel('time (a.u.)')
  ylabel({'normalized','FB weight'})
  if trialindex == 1
    t = title({['motor noise ',num2str(motornoisefactor)],['sensory noise ',num2str(sensorynoisefactor)]},'HorizontalAlignment','right');
    t.Units = 'normalized';
    t.Position(1) = 1;
  else
    title({num2str(motornoisefactor),num2str(sensorynoisefactor)})
  end

end
