import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNTEST(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(RNNTEST, self).__init__()
        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size
        
        self.rnn = nn.RNN(n_sensors,512,3,batch_first = True)      

        self.learn_dictionary = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(512*3, self.outputlayer_size)
            )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)    
    def forward(self, x):

        out,h = self.rnn(x)
        y = self.learn_dictionary(out)
        return y.unsqueeze(1)
        
class Sensor2PlotRNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, label_np, chosen_idx):
        train_set = torch.Tensor(data_np)
        self.train_set = train_set.flatten(2)[:, :, chosen_idx] # batch,seq,dim
        self.label_set = torch.Tensor(label_np).reshape(-1,1, 24*72)

    def __getitem__(self, index):
        return self.train_set[index], self.label_set[index]

    def __len__(self):
        return self.label_set.size(0)