from __future__ import print_function
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
from scipy.signal import cheb1ord, filtfilt, cheby1
end_point = 1375
Fs = 250. # sampling freq

def prepare_data_as(subj,runs,tw, cl=40,permutation=[47,53,54,55,56,57,60,61,62]):
    all_freqs = loadmat('./data/Freq_Phase.mat')['freqs'][0] # true freq
    step = 10
    ch = len(permutation) # # of channels
    x = np.array([],dtype=np.float32).reshape(0,tw,ch)
    y = np.zeros([0],dtype=np.int32) # true label
    file = loadmat('./data/S'+str(subj)+'.mat')['data']
    for run_idx in runs:
        for freq_idx in range(cl):
            raw_data = file[permutation,160:end_point,freq_idx,run_idx].T
            n_samples = 1  # the first data was used as the testing set
            _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
            _y = np.ones([n_samples],dtype=np.int32) * freq_idx
            for i in range(n_samples):
                _x[i,:,:] = raw_data[i*step:i*step+tw,:]
            x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl*samples
            y = np.append(y,_y)        # [?,1]
    x = filter(x)
    print('S'+str(subj)+'|x',x.shape)
    return x, y, all_freqs

def prepare_template(subj,runs,tw, cl=40,permutation=[47,53,54,55,56,57,60,61,62]):
    step = 10
    ch = len(permutation) # # of channels
    tr = len(runs)
    n_samples = 1  # the first data was used as the testing set
    template = np.zeros([n_samples,cl,tw,ch],dtype=np.float32)
    file = loadmat('./data/S'+str(subj)+'.mat')['data']
    for freq_idx in range(cl):
        raw_data = np.zeros([ch,end_point-160,tr],dtype=np.float32)
        for r in range(tr):
            raw_data[:,:,r] = file[permutation,160:end_point,freq_idx,runs[r]]
        # build template
        for i in range(n_samples):
            _t = np.zeros([ch,tw,tr],dtype=np.float32)
            for r in range(tr):
                _t[:,:,r] = raw_data[:,i*step:i*step+tw,r]
            _t = filter(_t) # filter 7 - 70 Hz
            template[i,freq_idx,:,:] = np.mean(_t,axis=-1).T
    template = np.tile(template, (cl,1,1,1)) # [cl*sample,cl,tw,ch]
    return template

## prepossing by Chebyshev Type I filter
def filter(x):
    nyq = 0.5 * Fs
    Wp = [6/nyq, 90/nyq];
    Ws = [4/nyq, 100/nyq];
    N, Wn=cheb1ord(Wp, Ws, 3, 40);
    b, a = cheby1(N, 0.5, Wn,'bandpass');
    # --------------
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = filtfilt(b,a,_x,padlen=3*(max(len(b),len(a))-1)) # apply filter
    return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## parameters
# channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
permutation = [47, 53, 54, 55, 56, 57, 60, 61, 62]
params = {'tw': 50, 'Fs': 250, 'cl': 40, 'ch': len(permutation)}

class multi_ch_Corr(nn.Module):
    def __init__(self, params, num, **kwargs):
        self.tw = params['tw']
        self.Fs = params['Fs']
        self.cl = params['cl']
        self.num = num
        self.corr = None
        super(multi_ch_Corr, self).__init__(**kwargs)
    def forward(self, input, **kwargs):
        x = input[0]  # [bs, tw * kernel_size_2]  signal
        x_ = torch.reshape(x, (-1, self.tw, self.num, 1))  # [bs, tw, kernel_size_2, 1]

        t = input[1]  # [bs, 1, tw * kernel_size_2, cl] reference
        t_ = torch.reshape(t, (-1, self.tw, self.num, self.cl))  # [bs, tw, kernel_size_2, cl]

        corr_xt = torch.sum(x_*t_, dim=1)  # [bs, kernel_size_2, cl]
        corr_xx = torch.sum(x_*x_, dim=1)  # [bs, kernel_size_2, 1]
        corr_tt = torch.sum(t_*t_, dim=1)  # [bs, kernel_size_2, cl]
        self.corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)  # [bs, kernel_size_2, cl]
        self.out = self.corr  # [bs, kernel_size_2, cl]
        self.out = torch.mean(self.out, dim=1)  # [bs, cl]
        return self.out

class ContrastiveLoss(nn.Module):
    def __init__(self, params):
        super(ContrastiveLoss, self).__init__()

    def forward(self, input, label, temperature):  # input: [bs, cl]   label: [bs]
        pos_mask = F.one_hot(label, num_classes=params['cl'])  # [bs, cl]
        pos_sim = torch.sum(input * pos_mask, dim=1)  # [bs]
        pos_sim = torch.exp(pos_sim/temperature)  # [bs]

        neg_mask = (torch.ones_like(pos_mask) - pos_mask).bool()
        neg_sim = input.masked_select(neg_mask).view(-1, params['cl']-1)  # [bs, cl-1]
        neg_sim = torch.exp(neg_sim/temperature)  # [bs, cl-1]
        neg_sim = torch.sum(neg_sim, dim=1)  # [bs]

        return (-torch.log(pos_sim / neg_sim)).mean()

class SiamCA(nn.Module):
    def __init__(self, params):
        super(SiamCA, self).__init__()
        self.kernel_size_1 = 9
        self.kernel_size_2 = 9
        self.lstm1 = nn.LSTM(input_size=params['ch'], hidden_size=self.kernel_size_1, batch_first=True)  # out = 32*50*10
        self.lstm2 = nn.LSTM(input_size=self.kernel_size_1, hidden_size=self.kernel_size_2, batch_first=True)  # 32*50*20
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(self.kernel_size_2*params['tw'])
        self.bn2 = nn.BatchNorm1d(self.kernel_size_2*params['tw'])
        self.corr = multi_ch_Corr(params=params, num=self.kernel_size_2)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), padding=(2, 0))

    def forward(self, input):
        sig = input[0]  # [bs,tw,ch]  = [100, 50, 9]
        ref = input[1]  # [bs, cl, tw, ch] = [100, 40, 50, 9]

        sig, _ = self.lstm1(sig)  # output size = [bs, tw, kernel_size_1]
        sig, _ = self.lstm2(sig)  # [bs, tw, kernel_size_2]
        sig = self.flatten1(sig)  # [bs, tw * kernel_size_2]
        sig = self.bn1(sig)  # [bs, tw * kernel_size_2]

        ref = torch.reshape(ref, (-1, params['tw'], params['ch']))  # [bs * cl, tw, ch]
        ref, _ = self.lstm1(ref)  # [bs*cl, tw, kernel_size_1]
        ref, _ = self.lstm2(ref)  # [bs*cl, tw, kernel_size_2]
        ref = torch.reshape(ref, [-1, params['tw']*self.kernel_size_2])  # [bs * cl, tw * kernel_size_2]
        ref = self.bn2(ref)  # [bs * cl, tw * kernel_size_2]
        ref = torch.reshape(ref, [-1, params['cl'], params['tw'] * self.kernel_size_2, 1])  # [bs, cl, tw * kernel_size_2, 1]
        ref = torch.transpose(ref, 1, 3)  # [bs, 1, tw * kernel_size_2, cl]
        corr = self.corr([sig, ref])  # [bs, cl]

        out = torch.reshape(corr, [-1, params['cl'], 1, 1])  # [bs, cl, 1, 1]
        out = torch.transpose(out, 1, 2)  # [bs, 1, cl, 1]
        out = self.conv(out)  # [bs, 1, cl, 1]
        out = torch.reshape(out, [-1, params['cl']])  # [bs, cl]
        return out, corr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


table = np.zeros((10, 3, 35), dtype=int)
table[0, 0, :] = [1, 4, 4, 3, 3, 4, 5, 5, 4, 3, 4, 2, 2, 2, 1, 4, 1, 3, 5, 4, 1, 2, 2, 1, 2, 2, 5, 3, 2, 5, 5, 5, 5, 3, 3]  # test0 validate1
table[0, 1, :] = [4, 2, 1, 1, 4, 5, 4, 4, 3, 2, 3, 3, 1, 3, 3, 3, 2, 1, 4, 3, 5, 1, 3, 3, 3, 1, 4, 2, 4, 3, 3, 2, 2, 2, 1]  # test0 validate2
table[0, 2, :] = [2, 3, 3, 4, 2, 2, 1, 1, 1, 4, 5, 4, 5, 4, 4, 5, 5, 4, 1, 5, 3, 4, 1, 2, 4, 4, 3, 5, 1, 2, 1, 1, 4, 5, 5]  # test0 validate3

table[1, 0, :] = [5, 3, 2, 2, 5, 5, 3, 4, 0, 2, 3, 5, 3, 0, 5, 0, 5, 5, 0, 3, 2, 3, 4, 3, 0, 5, 2, 0, 2, 5, 2, 4, 5, 2, 3]  # test1 validate1
table[1, 1, :] = [0, 5, 3, 3, 3, 3, 5, 2, 4, 3, 2, 3, 5, 5, 4, 3, 2, 3, 5, 2, 3, 4, 2, 0, 3, 2, 5, 3, 4, 2, 4, 2, 4, 4, 0]  # test1 validate2
table[1, 2, :] = [2, 2, 0, 5, 0, 0, 4, 3, 3, 5, 5, 2, 0, 2, 3, 2, 4, 0, 2, 0, 4, 2, 0, 2, 2, 0, 0, 4, 5, 0, 3, 3, 0, 3, 4]  # test1 validate3

table[2, 0, :] = [5, 1, 4, 0, 0, 3, 3, 5, 3, 3, 4, 4, 3, 0, 4, 4, 5, 5, 4, 1, 0, 1, 3, 0, 1, 4, 5, 1, 4, 5, 4, 3, 4, 1, 3]  # test2 validate1
table[2, 1, :] = [3, 5, 0, 4, 1, 5, 1, 4, 1, 1, 3, 5, 5, 3, 1, 5, 0, 4, 1, 3, 4, 3, 4, 1, 3, 1, 3, 4, 1, 4, 5, 1, 3, 5, 0]  # test2 validate2
table[2, 2, :] = [4, 3, 3, 3, 4, 4, 5, 3, 5, 0, 0, 0, 1, 5, 0, 3, 3, 3, 0, 5, 5, 0, 0, 4, 0, 5, 0, 5, 3, 3, 0, 0, 0, 4, 4]  # test2 validate3

table[3, 0, :] = [4, 5, 5, 1, 5, 2, 5, 4, 0, 1, 5, 4, 5, 5, 5, 2, 2, 4, 2, 2, 1, 5, 0, 1, 2, 5, 1, 1, 1, 2, 4, 5, 1, 0, 0]  # test3 validate1
table[3, 1, :] = [0, 2, 4, 5, 4, 0, 2, 0, 2, 0, 0, 1, 1, 1, 0, 5, 0, 2, 4, 4, 2, 0, 5, 4, 0, 0, 4, 0, 0, 1, 2, 1, 5, 5, 5]  # test3 validate2
table[3, 2, :] = [2, 0, 1, 0, 0, 4, 0, 1, 1, 5, 1, 5, 2, 4, 1, 1, 1, 5, 1, 1, 5, 2, 1, 0, 4, 2, 2, 5, 5, 0, 0, 2, 0, 1, 4]  # test3 validate3

table[4, 0, :] = [1, 1, 2, 0, 1, 1, 3, 5, 0, 2, 0, 1, 5, 1, 3, 1, 2, 2, 0, 1, 3, 3, 3, 2, 1, 0, 3, 3, 1, 1, 1, 0, 2, 1, 0]  # test4 validate1
table[4, 1, :] = [0, 5, 1, 1, 0, 5, 1, 1, 5, 1, 1, 3, 2, 0, 2, 5, 0, 5, 5, 5, 5, 5, 2, 3, 5, 2, 2, 2, 2, 2, 3, 2, 1, 3, 5]  # test4 validate2
table[4, 2, :] = [2, 0, 3, 3, 2, 3, 2, 3, 1, 0, 5, 2, 1, 2, 0, 2, 3, 3, 1, 2, 2, 0, 0, 0, 3, 5, 5, 1, 3, 3, 2, 3, 5, 5, 1]  # test4 validate3

table[5, 0, :] = [3, 2, 0, 4, 1, 0, 3, 0, 2, 0, 1, 4, 2, 2, 0, 0, 4, 0, 0, 2, 4, 1, 3, 2, 3, 3, 1, 2, 3, 2, 4, 3, 1, 0, 1]  # test5 validate1
table[5, 1, :] = [1, 4, 3, 2, 3, 2, 1, 3, 4, 3, 3, 3, 0, 1, 2, 1, 3, 3, 3, 0, 2, 0, 2, 1, 0, 4, 2, 0, 2, 0, 2, 4, 3, 3, 0]  # test5 validate2
table[5, 2, :] = [2, 1, 1, 3, 2, 1, 2, 4, 0, 4, 2, 0, 3, 0, 1, 4, 0, 1, 1, 3, 3, 3, 0, 3, 4, 1, 0, 1, 1, 4, 1, 1, 4, 2, 3]  # test5 validate3

best_vali_p = np.zeros((6, 35), dtype=int)
best_vali_n = np.zeros((6, 35), dtype=int)
# One of the three models that obtained the best classification result on the validation set was evaluated on the testing set.
#  0.2 s
best_vali_p[0,:] = [2, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 2, 2, 2, 1, 1, 0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 2, 1, 0, 0, 1, 1, 2, 0]
best_vali_p[1,:] = [2, 2, 0, 1, 1, 1, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 2, 1, 2]
best_vali_p[2,:] = [2, 0, 2, 2, 2, 2, 1, 2, 0, 2, 2, 2, 1, 0, 0, 2, 0, 2, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 2, 2, 2]
best_vali_p[3,:] = [2, 1, 1, 1, 1, 2, 1, 0, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 1, 0, 1, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
best_vali_p[4,:] = [2, 0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 1, 0, 0, 0]
best_vali_p[5,:] = [2, 0, 1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1]

best_vali_n[0,:] = [2, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 2, 2, 2, 1, 1, 0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 2, 1, 0, 0, 1, 1, 2, 0]
best_vali_n[1,:] = [2, 2, 0, 1, 1, 1, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 2, 1, 2]
best_vali_n[2,:] = [2, 0, 2, 2, 2, 2, 1, 2, 0, 2, 2, 2, 1, 0, 0, 2, 0, 2, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 2, 2, 2]
best_vali_n[3,:] = [2, 1, 1, 1, 1, 2, 1, 0, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 1, 0, 1, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
best_vali_n[4,:] = [2, 0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 1, 0, 0, 0]
best_vali_n[5,:] = [2, 0, 1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1]

acc_list = np.zeros((6, 35))
def main():
    for test_trial in range(0, 6):
        for subj in range(1, 36):
            test_run = [test_trial]
            x, y, __ = prepare_data_as(subj, test_run, params['tw'])  # [?,tw,ch]
            x = x.astype('float32')
            x = torch.Tensor(x)
            y = torch.Tensor(y).type(torch.LongTensor)

            train_run_1 = [0, 1, 2, 3, 4, 5]
            train_run_2 = [0, 1, 2, 3, 4, 5]
            validate_trial_1 = best_vali_p[test_trial, subj - 1]
            validate_trial_2 = best_vali_n[test_trial, subj - 1]
            validate_list_1 = table[test_trial, validate_trial_1, :]
            validate_list_2 = table[test_trial, validate_trial_2, :]

            print('test:', test_trial,'subj:', subj, 'validate1:', validate_trial_1, 'validate2:', validate_trial_2)

            validate_run_1 = [validate_list_1[subj-1]]
            validate_run_2 = [validate_list_2[subj - 1]]

            train_run_1.remove(test_run[0])
            train_run_1.remove(validate_run_1[0])
            train_run_2.remove(test_run[0])
            train_run_2.remove(validate_run_2[0])

            load_dir_1 = './50TimeWindow_positive/subj' + str(subj) + '_test' + str(
                test_run[0]) + '_validate' + str(validate_run_1[0]) +'.pth'
            load_dir_2 = './50TimeWindow_negative/subj' + str(subj) + '_test' + str(
                test_run[0]) + '_validate' + str(validate_run_2[0]) +'.pth'
            print('The best model will be loaded under path :  ' + load_dir_1 + load_dir_2)

            model_1 = SiamCA(params=params)
            model_1.load_state_dict(torch.load(load_dir_1))
            model_1.to(device)

            model_2 = SiamCA(params=params)
            model_2.load_state_dict(torch.load(load_dir_2))
            model_2.to(device)

            template_1 = prepare_template(subj, train_run_1, params['tw'])  # [cl*sample,cl,tw,ch]
            template_1 = template_1.astype('float32')
            template_1 = torch.Tensor(template_1)

            template_2 = prepare_template(subj, train_run_2, params['tw'])  # [cl*sample,cl,tw,ch]
            template_2 = template_2.astype('float32')
            template_2 = torch.Tensor(template_2)

            model_1.eval()
            model_2.eval()

            signal_1 = x.to(device).float()
            reference_1 = template_1.to(device).float()

            signal_2 = x.to(device).float()
            reference_2 = template_2.to(device).float()

            label = y.to(device)

            with torch.no_grad():
                out_1, _ = model_1([signal_1, reference_1])
                out_2, _ = model_2([signal_2, reference_2])

            output = out_1 + out_2  # ensemble the output of the two sides
            # calculate acc
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == label).sum().item()
            total = label.size(0)
            accuracy = float(correct) / total * 100
            print('acc:', accuracy)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    with torch.cuda.device(0):
        main()




