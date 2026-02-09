video_number = 308;
excel_file = sprintf('../data/intensities_total/%dN64intensities.xlsx', video_number);

% --- Read data ---
RI=readmatrix(excel_file, 'Sheet', 'Segment Intensities');
RI = RI(2:end, 2:end); % Drop headers
frames=1:length(RI);
N=size(RI,2); %segment count

seg=readmatrix(excel_file, 'Sheet', 'Anatomical Regions');
seg = seg(:, 2:end); % Drop headers
N1=seg(1,2); %end segment of JC; JC from 1 to N1
N2=seg(2,2); %edg segment of OT; OT from N1+1 to N2; SB from N2+1 to segment_count(N)

% flexfram=readmatrix(excel_file, 'Sheet', 'Flexion Frames');
% extfram=readmatrix(excel_file, 'Sheet', 'Extension Frames');
% Numcycle=size(flexfram,1)-1;

SRI=sum(RI,2); %total brightness of entire knee
% totoal brightness of three regions
SRI_JC=sum(RI(:,1:N1),2); SRI_OT=sum(RI(:,N1+1:N2),2); SRI_SB=sum(RI(:,N2+1:N),2);
% percentaged total brightness
pSRI_SB=SRI_SB./SRI; pSRI_OT=SRI_OT./SRI; pSRI_JC=SRI_JC./SRI;

figure; hold on
plot(frames, SRI, frames, SRI_JC, frames, SRI_OT, frames, SRI_SB)
legend('entire knee','JC', 'OT', 'SB');
xlabel('frames'); ylabel('total brightness')

figure; hold on
plot(frames, pSRI_JC, frames, pSRI_OT, frames, pSRI_SB)
legend('JC', 'OT', 'SB');
xlabel('frames'); ylabel('percentaged total brightness')

% -----Flux: calculation and smothing
p = 0.8; %Smoothing parameter p in [0,1]; p=1 interpolation; p=0 very smooth
pp_SB = csaps(frames, pSRI_SB, p);  % cubic smoothing spline
RISB_smooth = fnval(pp_SB, frames); % smoothed relative intensity for SB
pp_JC = csaps(frames, pSRI_JC, p); 
RIJC_smooth = fnval(pp_JC, frames);
% calculate flux 
fluxSB = fnder(pp_SB, 1); fluxSB_smooth = -fnval(fluxSB, frames); %from SB to OT
fluxJC = fnder(pp_JC, 1); fluxJC_smooth = fnval(fluxJC, frames); %from OT to JC
figure;hold on; grid on
plot(frames(1:end-1), -diff(pSRI_SB),'r:') % difference of SB array
plot(fluxSB_smooth,'r')
plot(frames(1:end-1), diff(pSRI_JC),'b:') % difference of JC array
plot(fluxJC_smooth,'b')
legend('raw flux SB->OT','smoothed', 'raw flux OT->JC', 'smoothed');
xlabel('frames')
ylabel('relative flux')


