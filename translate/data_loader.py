import torch
from torch.utils.data import Dataset, DataLoader

def get_from_file(file_path):
    data_zh = None
    data_en = None
    with open(file_path+".zh","r") as f:
        data_zh = f.read().split("\n")

    with open(file_path+".en","r") as f:
        data_en = f.read().split("\n")

    return data_en,data_zh


class MyDataset(Dataset):
    def __init__(self, data_en,data_zh,tokenizer_en,tokenizer_zh,max_len,padding):
        self.data_en = data_en
        self.data_zh = data_zh
        self.tokenizer_en = tokenizer_en
        self.tokenizer_zh = tokenizer_zh
        self.max_len = max_len
        self.padding = padding

        assert len(data_en) == len(data_zh)

    def __len__(self):
        return len(self.data_en)

    def __getitem__(self, index):
        data_en, data_zh = self.data_en[index],self.data_zh[index]
        data_en_ids,data_en_mask = self.tokenizer_en.encode(data_en,self.max_len,pad=True)
        data_zh_ids,data_zh_mask = self.tokenizer_zh.encode(data_zh,self.max_len,pad=True)

        return {
            "data_en_ids": torch.tensor(data_en_ids,dtype=torch.long),
            "data_en_mask": torch.tensor(data_en_mask, dtype=torch.long),
            "data_zh_ids": torch.tensor(data_zh_ids,dtype=torch.long),
            "data_zh_mask": torch.tensor(data_zh_mask, dtype=torch.long),
            "tgt":data_zh
        }