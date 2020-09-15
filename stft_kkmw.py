from numpy.ma import floor
import numpy as np
import numpy.matlib
def stft_kkmw(x,M,N,R,fs):

    num_frames = (floor((len(x)-M)/R+1))
    N_out= N/2+1
    nr_mics = len(x[0])

    xt = np.zeros((M, nr_mics))
    f_vec = np.arange(N/2+1) * fs/N
    t = np.arange(num_frames)*R/fs
    X=np.zeros((int(N_out),int(num_frames),int(nr_mics)),dtype=np.complex_)

    win = np.transpose(np.matlib.repmat(np.hamming(M), nr_mics, 1))

    for kk in range (0, int(num_frames)):
        xt=x[kk*R:kk*R+M]
        xt=xt*win
        Xt = np.fft.fft(xt, N,0)
        X[:,kk, 0:nr_mics] = Xt[0:int(N_out)]

    return X