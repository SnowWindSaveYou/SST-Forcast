from ast import parse
from networks.rnn import RNNTEST,Sensor2PlotRNNDataset
from networks import utils
from networks import utils_m
from networks import ssim
import setting

import os,argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="snn 0v0")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--targetm",type=int, default="12")
parser.add_argument("--pre_train_epochs",type=int, default=1500)
parser.add_argument("--fine_tune_epochs",type=int, default=500)


args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0')
RANDOM_SEED = 20
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LEAD_MON_TASKS = [0,1,3,6,9,12]# lead how many months
# LEAD_MON_TASKS = [3]
# LEAD_MON_TASKS = [0,1,6,9,12]

TARG_MON = args.targetm # the target month of forcast[1,12]
cmip_file_list = setting.CMIP_files
soda_file_list = setting.SODA_files
goda_file_list = setting.GODA_files
chosen_idx = setting.sensor_64_1
save_dir = "./models/"


model_name = "RNN_v2"
# model_name = "SNN_v2_ssim_L%d_T%d.pkl"

# pre train by CMIP
pre_train_batch = 20
pre_train_epochs = args.pre_train_epochs
pre_train_LR = 0.0003
pre_train_model_name = model_name+"_pt_L%d_T%d.pkl"

fine_tune_batch = 20
fine_tune_epochs = args.fine_tune_epochs
fine_tune_LR = 0.00003
fine_tune_model_name = model_name+"_ft_L%d_T%d.pkl"

mask = np.load("./datasets/mask.npy")
xmean_file = "./datasets/SODA_MON/soda_mean_m%d.npy"


model = RNNTEST(24*72,len(chosen_idx))
model = model.cuda()

# pre_train_file = "./models/SNN_v1_ft_L%d_T%d.pkl"
# model.load_state_dict(torch.load(save_dir+"RNNTEST_ft_L3_T12.pkl"))

for LEAD_MON in LEAD_MON_TASKS:# 12 mon

    # if os.path.exists(save_dir+model_name%(LEAD_MON,TARG_MON)):
    #     model.load_state_dict(torch.load(save_dir+model_name%(LEAD_MON,TARG_MON)))
    
    ## pre train with CMIP
    cmip_train_set,cmip_target_set = utils_m.load_dataset(LEAD_MON,TARG_MON,cmip_file_list,xmean_file)
    CMIP_Dataset = Sensor2PlotRNNDataset(cmip_train_set,cmip_target_set,chosen_idx )
    data_loader = DataLoader(
        dataset = CMIP_Dataset, 
        batch_size = pre_train_batch, 
        shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=pre_train_LR,weight_decay=1e-7)
    criterion = nn.MSELoss().cuda() 
    model, _ = utils.basicTrain(model,data_loader,optimizer,criterion,pre_train_epochs,save_dir+pre_train_model_name%(LEAD_MON,TARG_MON))
    # model, _ = ssim.SSIMTrain(model,data_loader,optimizer,pre_train_epochs,save_dir+pre_train_model_name%(LEAD_MON,TARG_MON))

    ## fine tuning with SODA
    soda_train_set,soda_target_set = utils_m.get_dataset_by_npy(LEAD_MON, TARG_MON, soda_file_list,xmean_file)
    SODA_Dataset = Sensor2PlotRNNDataset(soda_train_set*mask,soda_target_set*mask,chosen_idx )
    data_loader = DataLoader(
        dataset = SODA_Dataset, 
        batch_size = fine_tune_batch, 
        shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_LR,weight_decay=1e-7)
    criterion = nn.MSELoss().cuda() 
    model, _= utils.basicTrain(model,data_loader,optimizer,criterion,fine_tune_epochs,save_dir+fine_tune_model_name%(LEAD_MON,TARG_MON))
    # model, _= ssim.SSIMTrain(model,data_loader,optimizer,fine_tune_epochs,save_dir+fine_tune_model_name%(LEAD_MON,TARG_MON))

    torch.save(model.state_dict(), save_dir+model_name+"_L%d_T%d.pkl"%(LEAD_MON,TARG_MON))