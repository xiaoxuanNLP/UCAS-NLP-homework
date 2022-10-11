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


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=128*3, hidden_size=1024)
        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        x = einops.rearrange(x,'b c h w -> h b (c w)')
        output, (h, c) = self.lstm(x)
        out = self.linear(output[-1])
        out = torch.sigmoid(out)
        # print("out = ",out)
        return out

def train(save_path, epochs=3, print_step=50):
    model = LSTM()
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
            if step % print_step == 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                torch.save(model.state_dict(), save_path + '/e{}_s{}.pth'.format(epoch, step))
                print("epoch: {} , step: {} ".format(epoch, step))
                eval(LSTM(), save_path + '/e{}_s{}.pth'.format(epoch, step), test_loader, writer, epoch,
                     one_epoch_size, step)
            # print("output = ",output)


if __name__ == "__main__":
    train("LSTM", epochs=30, print_step=100)
