import torch
from torch.autograd import Variable
from .utils import subsequent_mask


def greedy_search(model,src,src_mask,cls=2,sep=3,max_len=64):
    memory = model.encode(src, src_mask)

    tgt = torch.ones(1,1).fill_(cls).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory,
                           src_mask,
                           Variable(cls),
                           Variable(subsequent_mask(tgt.shape[1])).type_as(src.data))
        prob = model.generator(out[:,-1])
        _,next_word = torch.max(prob,dim=1)
        next_word = next_word.data[0]
        if next_word == sep:
            break
        tgt = torch.cat([tgt,torch.ones(1,1).type_as(src.data).fill_(next_word)],dim=1)

    return tgt

def greedy_search_batch(model,src,src_mask,cls=2,sep=3,max_len=64):
    batch_size,src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    memory = model.encode(src,src_mask)
    tgt = torch.Tensor(batch_size,1).fill_(cls).type_as(src.data)

    for index in range(max_len):
        tgt_mask = subsequent_mask(tgt.shape[1]).expand(batch_size, -1, -1).type_as(src.data)
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)

        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            # print(stop_flag[i])
            if stop_flag[i] is False:
                if pred[i] == sep:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results