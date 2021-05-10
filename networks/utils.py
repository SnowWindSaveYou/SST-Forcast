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


def get_sensors(sample, sensor_size=64, miss_value=1e+20):
    sample = sample.flatten() != miss_value
    return np.random.choice(np.where(sample == True)[0], size=sensor_size)


def get_dataset_by_npy(LEAD_MON, TARG_MON, FILE_NAME):
    mm = num2mon(TARG_MON-LEAD_MON)
    mon_npy = [np.load(FILE_NAME % (num2mon(mm-i))) for i in range(3)]  # 倒序哒

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

    
    train_set[train_set>1000] = 0
    train_set[train_set<-1000] = 0
    target_set[target_set>1000] = 0
    target_set[target_set<-1000] = 0

    return train_set, target_set


def load_dataset(LEAD_MON, TARG_MON, file_list):
    train_list = []
    target_list = []
    for f in file_list:
        train_, target_ = get_dataset_by_npy(LEAD_MON, TARG_MON, f)
        train_list.append(train_)
        target_list.append(target_)
    train_set = np.concatenate(train_list, 0)
    target_set = np.concatenate(target_list, 0)

    return train_set, target_set


def show_heat_img(data, max_deg=35, min_deg=-5, center=15,
                  rev=True, cmap="rainbow", shape=(6, 2)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=shape)
    ax = sns.heatmap(data, vmax=max_deg, vmin=min_deg,
                     center=center, cmap=cmap)
    if rev:
        ax.invert_yaxis()

# class Sensor2PlotDataset(torch.utils.data.Dataset):
#     def __init__(self, data_np, label_np, chosen_idx):
#         train_set = torch.Tensor(data_np)
#         self.train_set = train_set.flatten(2)[:,:,chosen_idx].reshape(-1,3,8,8)
#         self.label_set = torch.Tensor(label_np).unsqueeze(1)

#     def __getitem__(self,index):
#         return self.train_set[index], self.label_set[index]
#     def __len__(self):
#         return self.label_set.size(0)


class Sensor2PlotDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, label_np, chosen_idx):
        train_set = torch.Tensor(data_np)
        self.train_set = train_set.flatten(
            2)[:, :, chosen_idx].reshape(-1, 1, 3*len(chosen_idx))
        self.label_set = torch.Tensor(label_np).reshape(-1, 1, 24*72)

    def __getitem__(self, index):
        return self.train_set[index], self.label_set[index]

    def __len__(self):
        return self.label_set.size(0)


def basicTrain(model, data_loader, optimizer, criterion, epochs=1000, save_dir=None):
    loss_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data.float(), requires_grad=True), Variable(
                target.float(), requires_grad=True)
            model.train()
            output = model(data)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

        cur_loss = running_loss/len(data_loader)
        loss_list.append(cur_loss)
        print('Epoche %s %s ' % (epoch, cur_loss))

        if (save_dir != None) and (epoch % 100 == 0):
            torch.save(model.state_dict(), save_dir)

    return model, loss_list


def img_validation(model, test_set):
    eval_model = model.eval().cuda()
    data, target = test_set[:]
    target = target.data.numpy()
    output = eval_model(data.cuda().float())
    output = output.cpu().data.numpy()
    nme = np.linalg.norm(target-output) / \
        np.linalg.norm(target)
    rmse = np.sqrt(np.var(output-target)/len(target))
    return rmse, nme
