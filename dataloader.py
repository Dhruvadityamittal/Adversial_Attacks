import torch
from torchvision import datasets, transforms

def get_data():
    train_mnist = torch.utils.data.DataLoader(datasets.MNIST("../MNIST_Data",train= True,download = True, transform = transforms.Compose([transforms.ToTensor()])),batch_size=32,shuffle=True)
    test_mnist = torch.utils.data.DataLoader(datasets.MNIST("../MNIST_Data",train= False,download = True, transform = transforms.Compose([transforms.ToTensor()])),batch_size=1,shuffle=True)
    return train_mnist, test_mnist