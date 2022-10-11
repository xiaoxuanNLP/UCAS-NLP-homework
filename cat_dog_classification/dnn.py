from data_loader import *
from config import *
from torch import cuda
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from eval import *
from torch.utils.tensorboard import SummaryWriter
import einops
import torch


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*128*128,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0),-1)
        output = self.feature(x)
        out = torch.sigmoid(output)
        # print("out = ",out)
        return out

def train(save_path, epochs=3, print_step=50):
    model = DNN()
    writer = SummaryWriter()

    train_set, test_set = load_data()
    train_set = MyDataset(train_set)
    test_set = MyDataset(test_set)

    train_dataloader_params = {
        "batch_size": CNN["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 1
    }

    test_dataloader_params = {
        "batch_size": CNN["VALID_BATCH_SIZE"],
        "num_workers": 1
    }

    train_loader = DataLoader(train_set, **train_dataloader_params)
    test_loader = DataLoader(test_set, **test_dataloader_params)

    device = 'cuda:0' if cuda.is_available() else 'cpu'

    model.to(device)

    train_param = {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005
    }

    optimizer = build_optimizer("SGD", model, **train_param)
    BCE = nn.BCELoss()

    one_epoch_size = len(train_loader)
    for epoch in range(epochs):
        for step, data in enumerate(train_loader, 0):
            model.train()
            optimizer.zero_grad()
            img, label = data["img"], data["label"]
            img_device = img.to(device)
            one_hot_label = F.one_hot(label, num_classes=2).to(device, dtype=torch.float)
            output = model(img_device)
            # print("output = ",output)
            # print("label = ",label)
            loss = BCE(output, one_hot_label)
            writer.add_scalar(tag="loss/train", scalar_value=loss.item(), global_step=epoch * one_epoch_size + step)

            loss.backward()

            optimizer.step()
            if step % print_step == 0 and step != 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                torch.save(model.state_dict(), save_path + '/e{}_s{}.pth'.format(epoch, step))
                print("epoch: {} , step: {} ".format(epoch, step))
                eval(DNN(), save_path + '/e{}_s{}.pth'.format(epoch, step), test_loader, writer, epoch,
                     one_epoch_size, step)
            # print("output = ",output)


if __name__ == "__main__":
    train("DNN", epochs=30, print_step=100)
