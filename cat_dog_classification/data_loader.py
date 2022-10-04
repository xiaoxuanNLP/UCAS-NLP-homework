import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random
from torchvision import datasets
import torchvision.transforms as transforms  # 这里打算不使用图片增强的策略
from PIL import Image

TRAIN_PATH = "./data/train/"
TEST_PATH = "./data/val/"


# 这里因为老师的数据并不是很多，直接全部放进内存中，加快读取速度
def load_data():
    data_transforms = {  # https://blog.csdn.net/weixin_41469023/article/details/123006025
        'train':
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    }

    train_list = os.listdir(TRAIN_PATH)
    test_list = os.listdir(TEST_PATH)

    train_datas = []
    test_datas = []
    for item in tqdm(train_list):
        label = 1 if "cat" in item else 0
        img = Image.open(TRAIN_PATH + item).convert('RGB')
        img = data_transforms['train'](img)
        train_datas.append((img, label))

    for item in tqdm(test_list):
        label = 1 if "cat" in item else 0
        img = Image.open(TEST_PATH + item).convert('RGB')
        img = data_transforms['val'](img)
        test_datas.append((img, label))

    random.shuffle(train_datas)
    random.shuffle(test_datas)
    return train_datas, test_datas


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img, label = self.datas[index]
        return {
            "img": img,
            "label": torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    load_data()
    # train_list_name = os.listdir("./data/train")
    # print("type(train_list_name) = ",type(train_list_name))
