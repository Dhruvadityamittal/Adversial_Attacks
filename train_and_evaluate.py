import torch 
import model 
from checkpoints import * 
import torch.nn.functional as F

def train_model(epochs, model,opt,train_mnist,device):
    for i in range(epochs):
        l = 0
        checkpoint = {'state_dict': model.state_dict() , 'opt': opt.state_dict()}
        save_checkpoint(checkpoint)
        for images,targets in train_mnist:
            images, targets = images.to(device), targets.to(device)
            outs = model(images)    
            opt.zero_grad()
            loss = F.nll_loss(outs,targets)
            loss.backward()
            opt.step()
        print("Epoch {}, Loss : {}".format(i,loss.item()))


def evaluate_model(model,test_mnist,device):
    model.eval()
    correct = 0 
    for data, target in test_mnist:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        
        pred = output.data.max(1)[1]      # Returns indices
        correct += pred.eq(target.data).cpu()

    print("Accuracy = ", correct.item()/len(test_mnist.dataset))
    