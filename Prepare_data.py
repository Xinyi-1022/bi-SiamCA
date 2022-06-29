from scipy.io import loadmat
import numpy as np
import math
from scipy.signal import cheb1ord, filtfilt, cheby1
end_point = 1375

Fs = 250. # sampling freq

################################################################################
# prepare training or test dataset
# input:
#       subj: subject index
#       runs: trial index
#       tw: time window length
#       cl: # of total frequency classes
#       permutations: channel indexes
# output:
#       x: dataset [?,tw,ch], ?=cl*runs*samples
#       y: labels [?,1]
#
def prepare_data_as(subj,runs,tw, cl=40,permutation=[47,53,54,55,56,57,60,61,62]):

    all_freqs = loadmat('./data/Freq_Phase.mat')['freqs'][0] # true freq
    step = 10 # 40ms
    ch = len(permutation) # # of channels
    x = np.array([],dtype=np.float32).reshape(0,tw,ch) # data
    y = np.zeros([0],dtype=np.int32) # true label
    file = loadmat('./data/S'+str(subj)+'.mat')['data']
    for run_idx in runs:
        for freq_idx in range(cl):
            raw_data = file[permutation,160:end_point,freq_idx,run_idx].T
            n_samples = int(math.floor((raw_data.shape[0]-tw)/step))
            _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
            _y = np.ones([n_samples],dtype=np.int32) * freq_idx
            for i in range(n_samples):
                _x[i,:,:] = raw_data[i*step:i*step+tw,:]

            x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl*samples
            y = np.append(y,_y)        # [?,1]

    x = filter(x)

    print('S'+str(subj)+'|x',x.shape)
    return x, y, all_freqs

################################################################################
# prepare reference signals
# input:
#       subj: subject index
#       runs: trial index
#       tw: time window length
#       cl: # of total frequency classes
#       permutations: channel indexes
# output:
#       template: reference signal [?,cl,tw,ch], ?=cl*samples
#
def prepare_template(subj,runs,tw,cl=40,permutation=[47,53,54,55,56,57,60,61,62]):
    step = 10 # 40ms
    ch = len(permutation) # # of channels
    tr = len(runs)
    n_samples = int(math.floor((end_point-160-tw)/step))
    template = np.zeros([n_samples,cl,tw,ch],dtype=np.float32)
    file = loadmat('./data/S'+str(subj)+'.mat')['data']
    for freq_idx in range(cl):
        raw_data = np.zeros([ch,end_point-160,tr],dtype=np.float32)
        for r in range(tr):
            raw_data[:,:,r] = file[permutation,160:end_point,freq_idx,runs[r]] # [ch,1250,runs]

        # build template
        for i in range(n_samples):
            _t = np.zeros([ch,tw,tr],dtype=np.float32)
            for r in range(tr):
                _t[:,:,r] = raw_data[:,i*step:i*step+tw,r]
            _t = filter(_t) # filter 7 - 70 Hz
            template[i,freq_idx,:,:] = np.mean(_t,axis=-1).T

    template = np.tile(template, (cl,1,1,1)) # [cl*sample,cl,tw,ch]


    return template # [cl,samples]



def normalize(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = (_x - _x.mean())/_x.std(ddof=1)
    return x



## prepossing by Chebyshev Type I filter
def filter(x):
    nyq = 0.5 * Fs
    Wp = [6/nyq, 90/nyq]
    Ws = [4/nyq, 100/nyq]
    N, Wn=cheb1ord(Wp, Ws, 3, 40)
    b, a = cheby1(N, 0.5, Wn,'bandpass')
    # --------------
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = filtfilt(b,a,_x,padlen=3*(max(len(b),len(a))-1)) # apply filter
    return x
