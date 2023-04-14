import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.conv2  = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self,x):

        conv1_out = F.relu(F.max_pool2d(self.conv1(x), 2))
        conv2_out = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(conv1_out)),2))
        
        conv2_out = conv2_out.view(-1,320)
        
        fc1_out = self.fc1(conv2_out)
        fc2_out = self.fc2(fc1_out)
       

        return F.log_softmax(fc2_out,dim=1)
