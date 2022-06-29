from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Prepare_data import prepare_data_as, prepare_template, normalize

import logging
import numpy as np
import time
import torch.nn.functional as F
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
        # self.contrastiveloss = ContrastiveLoss(params=params)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), padding=(2, 0))
        # self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), padding=(0, 0))

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

        # contrastive_loss = self.contrastiveloss(corr, label)
        out = torch.reshape(corr, [-1, params['cl'], 1, 1])  # [bs, cl, 1, 1]
        out = torch.transpose(out, 1, 2)  # [bs, 1, cl, 1]
        out = self.conv(out)  # [bs, 1, cl, 1]
        out = torch.reshape(out, [-1, params['cl']])  # [bs, cl]
        return out, corr

class prepare_data(Dataset):
    def __init__(self, subj, params, train_run, validate_run, test_run, split='train'):
        ## build signal and label
        if split == 'train':
            x, y, __ = prepare_data_as(subj, train_run, params['tw'])  # [?,tw,ch]
        elif split == 'validate':
            x, y, __ = prepare_data_as(subj, validate_run, params['tw'])  # [?,tw,ch]
        self.x = x  # [?,tw,ch]
        self.y = torch.Tensor(y).type(torch.LongTensor)
        template = prepare_template(subj, train_run, params['tw'])  # [cl*sample,cl,tw,ch]
        if split == 'train':
            self.template = np.tile(template, (len(train_run),1,1,1))
        elif split == 'validate':
            self.template = template
        self.x = self.x.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        signal = self.x[item]
        reference = self.template[item]
        label = self.y[item]
        return signal, reference, label

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# We give the three validate blocks we used in the following table. table[test, validate, subject]
table = np.zeros((6, 3, 35), dtype=int)
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

best_acc_list =[]
def main():
    batch_size = 100
    n_epochs = 300
    # The temperature should be calculated according to Eq.6 in our paper.
    # 0.1730, 0.0793, 0.0384, 0.0200, and 0.0117 for 0.2, 0.4, 0.6, 0.8, and 1.0 second data lengths, respectively.
    temperature = 0.17299
    save_model = False # True: save the model under your path.

    for test_trial in range(0, 6):
        for subj in range(1, 36):
            for validate_num in range(0,3):
                validate_trial = [table[test_trial, validate_num, subj-1]]
                print('subject:', subj, 'test_block:', test_trial, 'validate_block:', validate_trial[0])
                train_run = [0, 1, 2, 3, 4, 5]
                test_run = [test_trial]
                train_run.remove(test_run[0])
                train_run.remove(validate_trial[0])

                dataset = {split: DataLoader(
                    prepare_data(subj=subj, params=params, train_run=train_run, validate_run=validate_trial,
                                 test_run=test_run, split=split),
                    batch_size, shuffle=True, drop_last=True, num_workers=8) for split in ('train', 'validate')}

                save_dir = './50TimeWindow_negative/subj' + str(subj) + '_test' + str(test_run[0]) + '_validate' + str(validate_trial[0]) +'.pth'
                if save_model:
                    print('The best model will be saved under path :  ' + save_dir)
                else:
                    print('The model will not be saved.')

                model = SiamCA(params=params)
                model.to(device)
                criterion = nn.CrossEntropyLoss()  # The output loss have averaged by mini-batch; and do not need one-hot target
                criterion_Contra = ContrastiveLoss(params=params)
                optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999))
                max_validate_acc = 0.
                patient = 0
                for epoch in range(n_epochs):
                    start_time = time.time()
                    cl_losses = {'train': 0, 'validate': 0}
                    contrastive_losses = {'train': 0, 'validate':0}
                    acc = {'train': 0, 'validate': 0}
                    for split in ('train', 'validate'):
                        if split == 'train':
                            model.train()
                        else:
                            model.eval()
                        for i, batch in enumerate(dataset[split]):
                            signal = torch.flip(batch[0], dims=[1]).to(device).float()
                            reference = torch.flip(batch[1], dims=[2]).to(device).float()
                            label = batch[2].to(device)
                            if split == 'train':
                                output, corr = model([signal, reference])
                            else:
                                with torch.no_grad():
                                    output, corr = model([signal, reference])
                            # calculate loss
                            cl_loss = criterion(output, label)
                            cl_losses[split] += cl_loss.item()
                            contrastive_loss = criterion_Contra(corr, label, temperature=temperature)
                            contrastive_losses[split] += contrastive_loss
                            # calculate acc
                            _, predicted = torch.max(output.data, 1)
                            correct = (predicted == label).sum().item()
                            total = label.size(0)
                            accuracy = float(correct)/total
                            acc[split] += accuracy
                            if split =='train':
                                model.zero_grad()
                                loss = cl_loss + contrastive_loss
                                loss.backward()
                                clip_gradient(optimizer, 3.)  # gradient clipping
                                optimizer.step()

                    train_cl_loss = cl_losses['train']/len(dataset['train'])
                    validate_cl_loss = cl_losses['validate']/len(dataset['validate'])

                    train_contrastive_loss = contrastive_losses['train'] / len(dataset['train'])
                    validate_contrastive_loss = contrastive_losses['validate']/len(dataset['validate'])

                    train_acc = acc['train']/len(dataset['train'])*100
                    validate_acc = acc['validate']/len(dataset['validate'])*100

                    print('Epoch: {} - train_cl_loss: {:.4f} - val_cl_loss: {:.4f}'
                          '- train_contrastive_loss: {:.4f} - val_contrastive_loss: {:.4f}'
                          '- train_acc: {:.2f} - val_acc: {:.2f} - time:{:.2f}s'
                          .format(epoch, train_cl_loss, validate_cl_loss,
                                  train_contrastive_loss, validate_contrastive_loss,
                                  train_acc, validate_acc, (time.time() - start_time)))
                    # save best model's parameter
                    if validate_acc > max_validate_acc:
                        max_validate_acc = validate_acc
                        save_epoch = epoch
                        if save_model:
                            torch.save(model.state_dict(), save_dir)
                        patient = 0
                    else:
                        patient = patient + 1
                    if patient > 30:
                        break
                print('save_epoch', save_epoch, 'val_acc:', max_validate_acc)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    with torch.cuda.device(0):
        main()




