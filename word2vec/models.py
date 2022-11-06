import torch
import torch.nn as nn


class DM(nn.Module):
    def __init__(self, vec_dim, num_docs, num_words):
        super(DM, self).__init__()

        # 段的id
        self._Doc = nn.Parameter(
            torch.randn(num_docs+1, vec_dim), requires_grad=True
        )

        # 词矩阵
        self._Word = nn.Parameter(
            torch.randn(num_words+1,vec_dim), requires_grad=True
        )

        self._Output = nn.Parameter(
            torch.FloatTensor(vec_dim,num_words+1).zero_(),requires_grad=True # 这个地方注意要把输出初始化为0
        )

    def forward(self,context_ids,doc_ids,target_noise_ids):
        """
        tensor常用的高级切片：
        T[...,:n]   取最后1维的前n个元素
        T[:,:n]     取第2维的前n个元素
        T[:,:,:n]   取第3维的前n个元素
        T[:,:,:,:n] 取第4维的前n个元素
        """
        x = torch.add(
            self._Doc[doc_ids,:],torch.sum(self._Word[context_ids,:],dim=1)
        )
        # print("x = ",x.shape)
        # print("x.unsqueeze(1) = ",x.unsqueeze(1).shape)
        # print("self._Output[:,target_noise_ids].permute(1,0,2) = ",self._Output[:,target_noise_ids].permute(1,0,2).shape)
        # print("torch.bmm(x.unsqueeze(1),self._Output[:,target_noise_ids].permute(1,0,2)) = ",torch.bmm(x.unsqueeze(1),self._Output[:,target_noise_ids].permute(1,0,2)).shape)
        # print("torch.bmm(x.unsqueeze(1),self._Output[:,target_noise_ids].permute(1,0,2)).squeeze() = ",torch.bmm(x.unsqueeze(1),self._Output[:,target_noise_ids].permute(1,0,2)).squeeze().shape)
        return torch.bmm(x.unsqueeze(1),self._Output[:,target_noise_ids].permute(1,0,2)).squeeze()

    def get_word_vector(self,index):
        return self._Word[index,:].data.tolist()

    def get_paragraph_vector(self, index):
        return self._Doc[index,:].data.tolist()

if __name__ == "__main__":
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[2,3,4],[4,5,6],[7,8,9]])
    y = torch.tensor([1,2,3])
    print(x[y,:])