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


% format preliminary figure panels showing simulation results
numpanel = size(hd.crevecoeur.ax,1);
numtrial = size(hd.crevecoeur.ax,2);

for panel = 1:numpanel
  for trial = 1:numtrial
    xlim(panel,:,trial) = [min([hd.crevecoeur.ax(panel,trial).Children.XData]), ...
                           max([hd.crevecoeur.ax(panel,trial).Children.XData])]; %#ok<SAGROW>
    ylim(panel,:,trial) = [min([hd.crevecoeur.ax(panel,trial).Children.YData]), ...
                           max([hd.crevecoeur.ax(panel,trial).Children.YData])]; %#ok<SAGROW>
  end
end
joint.xlim = [min(xlim(:,1,:),[],3), max(xlim(:,2,:),[],3)];
joint.ylim = [min(ylim(:,1,:),[],3), max(ylim(:,2,:),[],3)];


% optionally, impose fixed x and y limits on the plot array for consistency
original.xlim = joint.xlim;
original.ylim = joint.ylim;
joint.ylim(4,:) = [0 1.7];    % normalised values
joint.ylim(3,:) = [0 .5];    % absolute values
paperxlim = [0.4 0.8];
for panel = 1:numpanel
  for trial = 1:numtrial
    hd.crevecoeur.ax(panel,trial).XLim = joint.xlim(panel,:);
    hd.crevecoeur.ax(panel,trial).YLim = joint.ylim(panel,:);
  end
end


% configure axis and grid visibility
for trial = 1:numtrial
  if trial == 1
    for panel = 2:4
      hd.crevecoeur.ax(panel,trial).XAxis.Visible = 'off';
    end
  else
    for panel = 1:4
      [hd.crevecoeur.ax(panel,trial).XAxis.Visible, ...
        hd.crevecoeur.ax(panel,trial).YAxis.Visible] = deal('off');
    end
  end
end
for trial = 1:numtrial
  for panel = 1:numpanel
    hd.crevecoeur.ax(panel,trial).YMinorGrid = 'on';
    hd.crevecoeur.ax(panel,trial).Box = 'off';
  end
end


% pick the desired RGB colours for subsequent plots
colour.scheme1 = [.7 .7 .7; ...
                  .5 .8 .5; ...
                  .2 .8 .4; ...
                  .15 .6 .3; ...
                  .1 .4 .2];
colour.scheme2 = [.6 .2 .4; ...
                  .9 .3 .5; ...
                  .9 .5 .7; ...
                  .8 .6 .7; ...
                  .7 .7 .7];


% prepare the final, "grid-shaped" figure showing parameter combinations
hd.joint.fg  = figure();
hd.joint.fg.Color = [1 1 1];
hd.joint.fg.Position = [3 33 1280 960];

fontsize = 13;

hnum = numel(motornoiselist);
vnum = numel(sensorynoiselist);


% copy graphs from preliminary (row-shaped) to final (grid-shaped) figure
for trial = 1:numtrial
  [vindex,hindex] = ind2sub([vnum hnum],trial);
  hd.joint.ax(vindex,hindex) = copyobj(hd.crevecoeur.ax(4,trial), hd.joint.fg);
  hd.joint.ax(vindex,hindex).XLim = paperxlim;
  hd.joint.ax(vindex,hindex).XTick = [];
end


% adjust the appearance of text
[hd.joint.ax(:,:).TitleFontSizeMultiplier, ...
 hd.joint.ax(:,:).LabelFontSizeMultiplier] = deal(1);
[hd.joint.ax(:,:).TitleFontWeight] = deal('bold');
for vindex = 1:vnum
  for hindex = 1:hnum
    hd.joint.ax(vindex,hindex).Children.Color = .4*[1 1 1];
    hd.joint.ax(vindex,hindex).YLabel.FontWeight = ...
      hd.joint.ax(vindex,hindex).TitleFontWeight;
    hd.joint.ax(vindex,hindex).YLabel.Visible = 'off';
    hd.joint.ax(vindex,hindex).Title.Visible = 'off';
    hd.joint.ax(vindex,hindex).YLabel.Color = colour.scheme1(vindex,:);
    hd.joint.ax(vindex,hindex).Title.Color = colour.scheme2(hindex,:);
    if hindex == 1
      hd.joint.ax(vindex,hindex).YLabel.Visible = 'on';
      hd.joint.ax(vindex,hindex).YLabel.String = ...
        {'sensory noise',[num2str(sensorynoiselist(vindex)),'x']};
    end
    if vindex == vnum
      hd.joint.ax(vindex,hindex).Title.Visible = 'on';
      hd.joint.ax(vindex,hindex).Title.String = ...
        ['motor noise ',num2str(motornoiselist(hindex)),'x'];
    end
  end
end


% choose which 1st panel to highlight with a bolder graph AND visible axes
vindex = vnum;
hindex = 1;

hd.joint.ax(1,hnum).XAxis.Visible = 'on';
hd.joint.ax(1,hnum).YAxis.Visible = 'on';
hd.joint.ax(1,hnum).YAxisLocation = 'right';
hd.joint.ax(1,hnum).YAxis.Label.Visible = 'on';
hd.joint.ax(1,hnum).YAxis.Label.String = '';
hd.joint.ax(1,hnum).YAxis.Label.FontWeight = 'normal';
hd.joint.ax(1,hnum).YAxis.Label.FontSize = hd.joint.ax(1,hnum).XAxis.Label.FontSize;
hd.joint.ax(1,hnum).YAxis.Label.Color = [0 0 0];

hd.joint.ax(vindex,hindex).Children.Color = [.0 .0 .0];
hd.joint.ax(vindex,hindex).Children.LineWidth = 2;
axes(hd.joint.ax(vindex,hindex))
hold on
hd.joint.ob = text(0,0,'');
hd.joint.ob.Units = 'normalized';
hd.joint.ob.HorizontalAlignment = 'left';
hd.joint.ob.VerticalAlignment = 'bottom';
hd.joint.ob.Position = [.0 0.8];
hd.joint.ob.FontSize = fontsize;
hd.joint.ob.FontWeight = 'bold';
hd.joint.ob.Color = hd.joint.ax(3,3).Children(end).Color;
hold off


% choose which 2nd panel to highlight with a bolder graph, but NOT axes
vindex = 1;
hindex = hnum;

hd.joint.ax(vindex,hindex).Children.Color = [.0 .0 .0];
hd.joint.ax(vindex,hindex).Children.LineWidth = 2;
axes(hd.joint.ax(vindex,hindex))
hold on
hd.joint.ob = text(0,0,'');
hd.joint.ob.Units = 'normalized';
hd.joint.ob.HorizontalAlignment = 'center';
hd.joint.ob.VerticalAlignment = 'bottom';
hd.joint.ob.Position = [.5 5.2];
hd.joint.ob.FontSize = fontsize;
hd.joint.ob.FontWeight = 'bold';
hd.joint.ob.Color = hd.joint.ax(3,3).Children(end).Color;
hold off


% set and assign panel dimensions
width = 170;
height = 120;
hsep = 20;
vsep = 20;
hanchor = 70;
vanchor = 240;

[vlist,hlist] = meshgrid(vanchor + (0:vnum-1)*(height+vsep), hanchor + (0:hnum-1)*(width+hsep));

for trial = 1:numtrial
  [vindex,hindex] = ind2sub([vnum hnum],trial);
  hd.joint.ax(vindex,hindex).Position = [hlist(hindex,vindex) vlist(hindex,vindex) width height];
end


% copy graphs to create overlays in the panels on the right and bottom
for hindex = 1:hnum
  
  hd.joint.ax(vnum+1,hindex) = axes();
  hd.joint.ax(vnum+1,hindex).Units = 'pixels';
  hd.joint.ax(vnum+1,hindex).Position = hd.joint.ax(1,hindex).Position - [0 210 0 0];
  hd.joint.ax(vnum+1,hindex).Visible = 'off';
  
  graphcounter = 0;
  for trial = vnum*(hindex-1) + (1:vnum)
    graphcounter = graphcounter + 1;
    hd.joint.sob(hindex,graphcounter) = ...
      copyobj(hd.crevecoeur.ax(4,trial).Children, hd.joint.ax(vnum+1,hindex));
  end

  for graph = 1:graphcounter
    hd.joint.sob(hindex,graph).Color = colour.scheme1(graph,:);
    hd.joint.sob(hindex,graph).LineWidth = 1.5;
  end
  
  hd.joint.ax(vnum+1,hindex).XLim = joint.xlim(4,:);
  hd.joint.ax(vnum+1,hindex).YLim = joint.ylim(4,:);
    
end

for vindex = 1:vnum
  
  hd.joint.ax(vindex,hnum+1) = axes();
  hd.joint.ax(vindex,hnum+1).Units = 'pixels';
  hd.joint.ax(vindex,hnum+1).Position = hd.joint.ax(vindex,hnum).Position + [265 0 0 0];
  hd.joint.ax(vindex,hnum+1).Visible = 'off';
  
  graphcounter = 0;
  for trial = vindex:hnum:numtrial
    graphcounter = graphcounter + 1;
    hd.joint.mob(hindex,graphcounter) = ...
      copyobj(hd.crevecoeur.ax(4,trial).Children, hd.joint.ax(vindex,hnum+1));
  end

  for graph = 1:graphcounter
    hd.joint.mob(hindex,graph).Color = colour.scheme2(graph,:);
    hd.joint.mob(hindex,graph).LineWidth = 1.5;
  end
  
  hd.joint.ax(vindex,hnum+1).XLim = joint.xlim(4,:);
  hd.joint.ax(vindex,hnum+1).YLim = joint.ylim(4,:);
  
end


% close the preliminary figure after all relevant content was copied
close(hd.crevecoeur.fg)


% add and configure panel lettering
hd.letter.ax = axes();
hd.letter.ax.Units = 'normalized';
hd.letter.ax.Position = [0 0 1 1];
hd.letter.ax.Units = 'pixels';
hd.letter.ax.Color = 'none';
hd.letter.ax.Visible = 'off';

letterfontsize = 15;
letterfontweight = 'bold';

hd.letter.ob(1) = text(10,940,'A','Units','pixels','FontSize',letterfontsize,'FontWeight',letterfontweight);
hd.letter.ob(2) = text(20,150,'B','Units','pixels','FontSize',letterfontsize,'FontWeight',letterfontweight);
hd.letter.ob(3) = text(1065,940,'C','Units','pixels','FontSize',letterfontsize,'FontWeight',letterfontweight);


% add dashed lines to indicate saccade onsets and offsets
empiricalsaccadeonset = .5;
empiricalsaccadeoffset = NaN;

for xpanel = 1:5
  for ypanel = 1:5
    hd.joint.ax(xpanel,ypanel).XLim = joint.xlim(4,:);
    hd.joint.ax(xpanel,ypanel).YLim = joint.ylim(4,:);
    
    axes(hd.joint.ax(xpanel,ypanel))
    hold on
    hd.joint.saccon(xpanel,ypanel)  = plot(empiricalsaccadeonset*[1 1],  hd.joint.ax(xpanel,ypanel).YLim, '--', 'Color', .5*[1 1 1]);
    hd.joint.saccoff(xpanel,ypanel) = plot(empiricalsaccadeoffset*[1 1], hd.joint.ax(xpanel,ypanel).YLim, '--', 'Color', .5*[1 1 1]);
    hold off
    
  end
end

for xpanel = 6
  for ypanel = 1:5
            
    axes(hd.joint.ax(xpanel,ypanel))
    hold on
    hd.joint.saccon(xpanel,ypanel)  = plot(empiricalsaccadeonset*[1 1],  hd.joint.ax(xpanel,ypanel).YLim, '--', 'Color', .5*[1 1 1]);
    hd.joint.saccoff(xpanel,ypanel) = plot(empiricalsaccadeoffset*[1 1], hd.joint.ax(xpanel,ypanel).YLim, '--', 'Color', .5*[1 1 1]);
    hold off
            
  end
end

for xpanel = 1:5
  for ypanel = 6
            
    axes(hd.joint.ax(xpanel,ypanel))
    hold on
    hd.joint.saccon(xpanel,ypanel)  = plot(empiricalsaccadeonset*[1 1],  hd.joint.ax(xpanel,ypanel).YLim, '--', 'Color', .5*[1 1 1]);
    hd.joint.saccoff(xpanel,ypanel) = plot(empiricalsaccadeoffset*[1 1], hd.joint.ax(xpanel,ypanel).YLim, '--', 'Color', .5*[1 1 1]);
    hold off

  end
end


% impose consistent limits and ticks
for k = 1:numel(hd.joint.ax)
  if isgraphics(hd.joint.ax(k))
    hd.joint.ax(k).XLim = [0.35 0.8];
    hd.joint.ax(k).XTick = [];
    hd.joint.ax(k).Position(2) = hd.joint.ax(k).Position(2) - 10;
  end
end
hd.joint.ob.Position(2) = .5;


% create a separate figure overlaying the macaque- and zebrafish-like plots
hd.overlay.fg = figure();
hd.overlay.fg.Color = [1 1 1];
hd.overlay.fg.Position = [200 200 400 300];

fontsize = 13;

hd.overlay.ax = axes();
hd.overlay.ax.XLim = hd.joint.ax(1,1).XLim;
hd.overlay.ax.YLim = hd.joint.ax(1,1).YLim;

hd.overlay.fsh = copyobj(hd.joint.ax(4,4).Children, hd.overlay.ax);
hd.overlay.mac = copyobj(hd.joint.ax(5,1).Children([1,2,4]), hd.overlay.ax);
hd.overlay.fsh = copyobj(hd.joint.ax(1,5).Children, hd.overlay.ax);

hd.overlay.ax.YLim = [0 1.6];
hd.overlay.ax.XLabel = hd.joint.ax(1,5).XLabel;
hd.overlay.ax.YLabel = hd.joint.ax(1,5).YLabel;
hd.overlay.ax.Title.String = 'inter-species predictions';
hd.overlay.ax.Title.Visible = 'off';
hd.overlay.mac(3).Color = [1 .7 .3];
hd.overlay.fsh(2).Color = [.2 .5 .8];
hd.overlay.mac(2).Visible = 'off';
hd.overlay.fsh(4).Color = .8*[1 1 1];
hd.overlay.fsh(4).LineWidth = 1.5;  
hd.overlay.fsh(4).LineStyle = '-';
uistack(hd.overlay.fsh(2),'bottom')
hd.overlay.ax.Units = 'pixel';
hd.overlay.ax.Position = [53 44 340 234];
hold on
hd.overlay.mac(end+1) = plot([0.70,0.7125], 0.57*[1 1], 'Color', hd.overlay.fsh(4).Color, 'LineWidth', 2);
hd.overlay.fsh(end+1) = plot([0.70,0.75], 0.53*[1 1], 'Color', hd.overlay.mac(3).Color, 'LineWidth', 2);
hd.overlay.fsh(end+1) = text(0.725, 0.51, 'e.g., 10 msec', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontWeight', 'bold');
hd.overlay.fsh(end+1) = text(0.49, 1.30, {'saccade','onset'}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', hd.overlay.fsh(2).Color, 'FontWeight', 'bold');
hd.overlay.fsh(end+1) = plot(0.405*[1 1], [0.18, 0.68], 'Color', [0 0 0], 'LineWidth', 2);
hd.overlay.fsh(end+1) = text(0.395, 0.43, '50%', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'Color', [0 0 0], 'FontWeight', 'bold');
hold off

hd.joint.ax(1,3).XLabel = hd.overlay.ax.XLabel
hd.joint.ax(1,3).YLabel = hd.overlay.ax.YLabel

hd.joint.fg.Position = [50 50 1300 940]

hd.overlay.ax.Visible = 'off';


% fix minor visual bug resulting from MATLAB version differences
for motorindex = 1:numel(motornoiselist)
  for sensorindex = 1:numel(sensorynoiselist)
    hd.joint.ax(motorindex,sensorindex).Color = [1 1 1];
  end
end


% finalise appearance of the hightlighted panel on the bottom right
hd.joint.ax(1,5).XColor = 'k';
hd.joint.ax(1,5).YColor = 'k';
hd.joint.ax(1,5).YMinorGrid = 'off';
hd.letter.ob(1).Color = 'k';
hd.letter.ob(2).Color = 'k';
hd.letter.ob(3).Color = 'k';
hd.joint.ax(1,5).TickLength = [0.03 0.075];
hd.joint.ax(1,5).TickDir = 'out';
hd.joint.ax(1,5).YLabel.String = '1st Kálmán component';
hd.joint.ax(1,5).YLabel.Position = [0.91 0.85 -1];