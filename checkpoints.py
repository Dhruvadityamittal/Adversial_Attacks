import torch 

def save_checkpoint(state, file_name = "mnist.pth.tar"):
    torch.save(state,file_name)

def load_checkpoint(model,opt, checkpoint):
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['opt'])
