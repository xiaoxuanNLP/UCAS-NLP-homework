import torch
import torch.optim as optim

def build_optimizer(optim_name,model,**param):
    if optim_name == "SGD":
        return optim.SGD(model.parameters(),lr=param["learning_rate"],momentum=param["momentum"],weight_decay=param["weight_decay"])