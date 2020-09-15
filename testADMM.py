import time
import scipy.io
import soundfile as sf
import numpy as np
from ADMM import ADMM
from stft_kkmw import stft_kkmw

Nmic = 4
winsize = 320
winshift = 160
data = np.empty([127523, Nmic])
K = 512

for v in range(1, Nmic + 1):
    [data[:, v - 1], fs] = sf.read('testFiles/AMI_WSJ20-Array1-' + str(v) + '_T10c0201.wav')

# Signal transformation into STFT domain
xk = np.moveaxis(stft_kkmw(data, winsize, K, winshift, fs), 0, 1)

# Time test
PythonADMMProcessingTime = 0
#for k in range(1, 101):
startTime = time.time()
[dk, ck] = ADMM(xk, num_mic=Nmic)
stopTime = time.time()
PythonADMMProcessingTime += stopTime - startTime
#PythonADMMProcessingTime /= 100

MatlabADMMProcessingTime = scipy.io.loadmat('MatlabADMMTime.mat')['MatlabADMMProcessingTime']
ckMatlab = scipy.io.loadmat('ckMatlab.mat')['ck']
dkMatlab = scipy.io.loadmat('dkMatlab.mat')['dk']
cktest = np.amax(ck - ckMatlab)
dktest = np.amax(dk[:, :, 0] - dkMatlab)

print("Ck test: ", cktest)
print("Dk test: ", dktest)
print("Matlab processing time: ", MatlabADMMProcessingTime) #Result: 3.68429289
print("Python processing time: ", PythonADMMProcessingTime) #Result: 9.127202084064484
