import torch
import torch.nn as nn
import torch.optim as optim

import model 
from train_and_evaluate import train_model, evaluate_model
from checkpoints import * 

from Attack import *
from dataloader import * 
from plot_graphs import * 

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

train_mnist ,test_mnist = get_data()

plot_orignal(train_mnist)

model = model.Net().to(device)
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

load_check = True
epochs = 10 

if(load_check):
    load_checkpoint(model,opt,torch.load(r'C:\Users\Dhruv\OneDrive\Desktop\My_projects\Adversial_Attacks\mnist.pth.tar') )
else:    
    train_model(epochs,model,opt,train_mnist,device)


evaluate_model(model,test_mnist,device)

accuracies = []
examples = []

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/mnist.pth.tar"

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_mnist, eps)
    accuracies.append(acc)
    examples.append(ex)


# Plot several examples of adversarial samples at each epsilon
plot_pertubations(epsilons,examples)