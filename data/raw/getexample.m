xcld = 200;
minLen = 1000;
prelen = 50;
forward = 10;
filt_xcld = 50;
rng(111);

fs = 1000;
oscBand=[30,80];
[bFilt, aFilt] = butter(2,oscBand/(fs/2));

load('subjects.mat');
LFP = subject{4};
Y = fft(LFP);
T = length(LFP);
seg = getseg(LFP,xcld,minLen);
nseg = size(seg,2);
tseg = seg(2,:)-seg(1,:)-prelen-forward+1-filt_xcld*2;
tcum = cumsum(tseg);

iseg = ones(1,tcum(end));
idx = false(1,T);
idx(seg(1,1)+1:seg(2,1)) = true;
for i = 2:nseg
    iseg(tcum(i-1)+1:tcum(i)) = i;
    idx(seg(1,i)+1:seg(2,i)) = true;
end

ntrain = round(T/prelen);
t = sort(randperm(tcum(end),ntrain));
ts = zeros(1,ntrain);
for i = 1:ntrain
    id = iseg(t(i));
    ts(i) = t(i)-tcum(id)+tseg(id)+seg(1,id)+filt_xcld;
end

%ZS = zscore(LFP);
ZS = (LFP-mean(LFP(idx)))/std(LFP(idx));
ZS_gamma = filtfilt(bFilt,aFilt,ZS);
hb_gamma = hilbert(ZS_gamma);
amp_gamma = abs(hb_gamma);

x = zeros(ntrain,prelen+1);
for i = 1:ntrain
    x(i,1:prelen) = amp_gamma(ts(i):ts(i)+prelen-1);
    x(i,end) = amp_gamma(ts(i)+prelen+forward-1);
end
csvwrite('train_sub4.csv',x);
