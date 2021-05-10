import torch
import torch.nn as nn
import torch.nn.functional as F

class shallow_decoder(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(shallow_decoder, self).__init__()
        
        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size
        
        self.learn_features = nn.Sequential(         
            nn.Linear(n_sensors, 350),
            nn.ReLU(True), 
            nn.BatchNorm1d(1),  
            nn.Dropout(0.1)
            )        
        
        self.learn_coef = nn.Sequential(            
            nn.Linear(350, 450),
            nn.ReLU(True),  
            nn.BatchNorm1d(1),  
            nn.Linear(450, 1200),
            nn.ReLU(True),  
            nn.BatchNorm1d(1),  
            )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(1200, self.outputlayer_size),
            )
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)    


    def forward(self, x):
        x = self.learn_features(x)
        x = self.learn_coef(x)
        x = self.learn_dictionary(x) 
        return x