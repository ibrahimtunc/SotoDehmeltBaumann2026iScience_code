function [x,control,error,L,K,cost] = NoiseComparison_ControlModel(xinit,xfinal,simdata,p) % FD20220410

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
  % Code in the present file almost exclusively stems from prior work done 
  % by Frederic Crevecoeur in conjunction with their previous publication,
  % 
  % Frédéric Crevecoeur, Konrad P Kording (2017) Saccadic suppression as a 
  % perceptual consequence of efficient sensorimotor estimation. 
  % eLife 6:e25073, https://doi.org/10.7554/eLife.25073 
  % 
  % We thank F.C. for sharing his code with us, and gratefully acknowledge 
  % his support in reproducing the findings or their earlier paper.
  

  delta = simdata.delta;
  fixation = simdata.fixation;
  time = simdata.timefree;
  stab = simdata.timestab;
  delay = simdata.delay;
  alpha = simdata.alpha; 
  alphaNoisec = simdata.alphac;
  alphaDiscrete = alphaNoisec*delta^-0.5;  
  nStep = round((fixation+time+stab)/delta);
  
  
  A = [0 1 0 0 0 0;-1/(.013*.224) -(.013+.224)/(.013*.224) 0 0 0 0];
  A = [A;zeros(4,6)];
  A(2,1) = 0;   % FD20220410: deactivate drift to centre!!!
  A(end-1,end) = 1;
  B = [0;1/(.13*.224);0;0;0;0];
  BSDN = [0;1/(.13*.224);0;1/(.13*.224);0;1/(.13*.224)]; %#ok<NASGU>
  
  Acont = A;      % continusous state space representation matrices
  Bcont = B;
  
  n = size(A,1);
  A0 = expm(delta*A);
  B0 = discreteB(B,A,delta);
  BSDN0 = [B0(1:2);B0(1:2);B0(1:2)]; % parameter for signal dependent noise
  
  H0 = eye(n);        
  h = round(delay/delta);
  nh = size(H0,1);
  
  % Cost parameters
  Q0 = zeros(n,n,nStep+1);
  Qfinal = alpha(1)^-1*([1;0;0;0;-1;0]*[1 0 0 0 -1 0] + 0*[0;1;0;0;0;-1]*[0 1 0 0 0 -1]);
  Qfixation = alpha(1)^-1*([1;0;-1;0;0;0]*[1 0 -1 0 0 0] + 0*[0;1;0;-1;0;0]*[0 1 0 -1 0 0]);
  nMove = round((time+fixation)/delta);
  
  % Second fixation
  for i = nMove:nStep+1   
    Q0(:,:,i) = Qfinal;      
  end
  
  % first fixation
  nFixation = round(fixation/delta);
  
  for i = 1:nFixation
    Q0(:,:,i) = Qfixation;
  end
      
  % control cost
  R = zeros(size(B,2),size(B,2),nStep);
  
  for i = 1:nStep
    R(:,:,i) = 10^-2*eye(size(B,2));      
  end
    
  % other noise parameters
  oZeta = zeros(n);
  [A,B,H,~] = augmentSystem(A0,B0,H0,oZeta,h,0);
  C = alphaDiscrete*B; 
  C(2:3) = C(1:2);
  C(3:4) = C(1:2);
  Q = augmentConstraints2(Q0,h,nStep+1); %#ok<NASGU>
  
  % initialisation matrices
  ce = simdata.noise(1)*eye(size(A));
  
  % initial condition
  x0 = [xinit;xinit;xfinal];
  z0 = zeros(n*(h+1),1);
  
  for i = 1:h+1
    z0((i-1)*n+1:i*n) = x0;  
  end
  
  oOmega = simdata.noise(2)*eye(nh);
  oXi = simdata.noise(3)*B*B';
  ceta = oXi;  
  
  % calculate basic feedback control gains
  [L_new,~] = basicLQG(A0,B0,Q0,R,x0,oZeta);
  
  xallNew = zeros(p,nStep+1);
  xallEstNew = zeros(p,nStep+1);
  vallNew = zeros(p,nStep+1);
  vallEstNew = zeros(p,nStep+1);
  tallNew = zeros(p,nStep+1);
  tallEstNew = zeros(p,nStep+1);
  
  Abuffer = zeros(h+1);
  Abuffer(2:end,1:end-1) = eye(h);
  
  controlClassic = zeros(p,nStep);
  controlNew = controlClassic;
  control_noise = zeros(p,nStep);
  
  SuppIndex = zeros(p,nStep);
  SuppIndexClassic = SuppIndex;
  SigmayyOut = zeros(6,6,nStep);
  SigmayyOutClassic = zeros(6,6,nStep);
  Vout = zeros(2,nStep);
 
  ce0 = ce;
  cost = zeros(p,2);
  
  for j = 1:p
      
    ce = ce0;
    Sigmaxx = ce(1:size(A0,1),1:size(A0,2));
    noiseInit = mvnrnd(zeros(size(z0,1),1),ce0)';  
    currentEstimateNew = x0+0*noiseInit(1:6);
    currentStateNew = z0;
    
    uBuffer = ones(1,h+1)*xinit(1);
          
    for i = 1:nStep
            
      % adaptive LQG with delays
      xallNew(j,i) = currentStateNew(1);
      xallEstNew(j,i) = currentEstimateNew(1);
      vallNew(j,i) = currentStateNew(2);
      vallEstNew(j,i) = currentEstimateNew(2);
      tallNew(j,i) = currentStateNew(5);
      tallEstNew(j,i) = currentEstimateNew(5);       
      
      uNew = -L_new(:,:,i)*currentEstimateNew;
      
      cost(j,1) = cost(j,1) + ...
        currentStateNew(1:6)'*Q0(:,:,i)*currentStateNew(1:6) + ...
        uNew'*R(:,:,i)*uNew;
      
      uBuffer = uBuffer*Abuffer;
      uBuffer(end)=uNew;
      controlNew(j,i)=controlNew(j,i)+uNew;

      % dynamics and feedback
      nc = size(C,3);
      sdn = 0;
      
      for k = 1:nc
        eps = normrnd(0,1);
        sdn = sdn+eps*C(:,:,k)*uNew;
      end
      
      addN = mvnrnd(zeros(n*(h+1),1),oXi)';
      nextStateNew = A*currentStateNew + B*uNew + sdn + addN;
      y = H*currentStateNew + mvnrnd(zeros(nh,1),oOmega)';
      
      control_noise(j,i) = sdn(1)+addN(1);
      
      % extrapolation of sensory feedback
      [xy,V,M] = integrate(y,delta,delay,uBuffer,Acont,Bcont,'continuous',1);

      % adding the additive noise to the integrated SDN
      V(1:2,1:2) = alphaNoisec^2*V(1:2,1:2)+(delay/delta)*oXi(1:2,1:2);
      
      % SDN everywhere
      V(3:4,3:4) = V(1:2,1:2);
      V(5:6,5:6) = V(1:2,1:2);
              
      priorNew = A0*currentEstimateNew + B0*uNew + ...
        mvnrnd(zeros(n,1),ceta(1:n,1:n))' ;
      Ve = alphaDiscrete*BSDN0*uNew^2*BSDN0'+oXi(1:n,1:n);
      Vout(1,i) = norm(V(1,1));
      Vout(2,i) = Ve(1,1);
      Sigmaxy = A0*Sigmaxx;
      Sigmayy = Sigmaxx + M*oOmega*M' + V;
      if sum(abs(Sigmayy)) == 0
        placeholder = zeros(size(Sigmaxy));
      else  
        placeholder = Sigmaxy/Sigmayy;
      end
      Sigmaxx = A0*Sigmaxx*A0'+Ve+ceta(1:n,1:n)-placeholder*Sigmaxy';
     
      nextEstimateNew = priorNew + placeholder*(xy-currentEstimateNew);
      
      % iterate index
      currentStateNew = nextStateNew;
      currentEstimateNew = nextEstimateNew;
      
      modKalman = placeholder;
      SuppIndex(j,i) = SuppIndex(j,i)+norm(modKalman(1,1));
      SigmayyOut(:,:,i) = Sigmayy;
                  
    end
    
    xallNew(j,end) = currentStateNew(1);
    xallEstNew(j,end) = currentEstimateNew(1);
          
  end

    L.Ln = L_new;
    x.xn = xallNew;
    x.vn = vallNew;
    x.tn = tallNew;
    x.xen = xallEstNew;
    x.ven = vallEstNew;
    x.ten = tallEstNew;
    
    control.cn = controlNew/p;
    
    error = 0;
    K.index = SuppIndex;
    K.indexClassic = SuppIndexClassic;
    K.Sigmayy = SigmayyOut;
    K.SigmayyClassic = SigmayyOutClassic;
    K.Vout = Vout;
      
end

