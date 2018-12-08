Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 3600000;          % Length of signal
t = (0:L)*T;        % Time vector

load('subjects.mat');
X = subject{4};
XX = X;
%csvwrite('rawr_sub4.csv',XX);

Y = fft(X);
spectrogram(X,[],[],[],Fs,'yaxis');colorbar;

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
%plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%plot(t',Y)