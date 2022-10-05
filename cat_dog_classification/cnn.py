from data_loader import *
from config import *
from torch import cuda
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from eval import *
from torch.utils.tensorboard import SummaryWriter


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(13 * 13 * 128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        # print("input.shape = ",x.shape)
        x = self.features(x)
        # print("x.shape = ",x.shape)
        x = torch.flatten(x, start_dim=1)
        # print("x.flatten = ",x.shape)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x


def train(save_path, epochs=3, print_step=50):
    model = AlexNet()
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

                eval(AlexNet(), save_path + '/e{}_s{}.pth'.format(epoch, step), test_loader, writer, epoch,
                     one_epoch_size, step)
            # print("output = ",output)


if __name__ == "__main__":
    train("AlexNet",epochs=30, print_step=5)
