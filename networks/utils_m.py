import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def num2mon(n):
    m = n % 12
    if m == 0:
        return 12
    else:
        return int(m)


def get_lead_year(TARG_MON, LEAD_MON):
    lead = TARG_MON - LEAD_MON
    if lead > 0:
        return 0
    else:
        return int(-lead/12)+1
def get_dataset_by_npy(LEAD_MON, TARG_MON, FILE_NAME,MEAN_FILE_NAME=None):
    mm = num2mon(TARG_MON-LEAD_MON)
    mon_npy = [np.load(FILE_NAME%( int(num2mon(mm-i)) )) for i in range(3)]  # 倒序哒

    # ano
    if MEAN_FILE_NAME is not None:
        mon_mean_npy = [np.load(MEAN_FILE_NAME%(num2mon(mm-i)) ) for i in range(3)] 
        for i in range(3):
            mon_npy[i] = mon_npy[i]-mon_mean_npy[i]
    else:
        for i in range(3):
            mon_npy[i] = mon_npy[i]-mon_npy[i].mean(0)

    lead_years = []
    max_lead_year = 0
    for i in range(3):
        lead_year = get_lead_year(TARG_MON, LEAD_MON+i)
        max_lead_year = max(max_lead_year, lead_year)
        lead_years.append(lead_year)

    train_set = []
    for i in range(3):
        if lead_years[i] > 0:
            lead = max_lead_year-lead_years[i]
            train_set.append(mon_npy[i][lead: -lead_years[i]])
        else:
            train_set.append(mon_npy[i][max_lead_year:])

    train_set = np.asarray(train_set)
    train_set = np.swapaxes(train_set, 0, 1)  # 交换月份维度,输出(B,M=3,H=24,W=72)

    target_set = np.load(FILE_NAME % TARG_MON)[max_lead_year:]

    #ano
    if MEAN_FILE_NAME is None:
        target_set -= target_set.mean(0)
    else:
        target_set -= np.load(MEAN_FILE_NAME%TARG_MON )
    
    train_set[train_set>1000] = 0
    train_set[train_set<-1000] = 0
    target_set[target_set>1000] = 0
    target_set[target_set<-1000] = 0

    return train_set, target_set

def load_dataset(LEAD_MON, TARG_MON, file_list, mean_file=None):
    train_list = []
    target_list = []
    for i in range(len(file_list)):
        if mean_file is not None:
            train_, target_ = get_dataset_by_npy(LEAD_MON, TARG_MON, file_list[i], mean_file)
        else:
            train_, target_ = get_dataset_by_npy(LEAD_MON, TARG_MON, file_list[i])
        train_list.append(train_)
        target_list.append(target_)
    train_set = np.concatenate(train_list, 0)
    target_set = np.concatenate(target_list, 0)

    return train_set, target_set


class Sensor2PlotWithdrawDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, label_np, mask, chosen_idx):
        self.mask = mask
        train_set = torch.Tensor(data_np)
        self.train_set = train_set.flatten(
            2)[:, :, chosen_idx].reshape(-1, 1, 3*len(chosen_idx))
        self.label_set = torch.Tensor(label_np).reshape(-1, 1, 24*72)[:,:,mask.flatten()!=0]

    def __getitem__(self, index):
        return self.train_set[index], self.label_set[index]

    def __len__(self):
        return self.label_set.size(0)
    def reform(self,x):
        y = np.zeros((len(x),24*72))
        y[:,self.mask.flatten()!=0] = x[:,0,:]
        return y