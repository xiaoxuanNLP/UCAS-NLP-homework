import torch
from torch.optim import Adam
from data import *
from torch.utils.data import DataLoader
from torch import cuda
from models import *
from loss import NegativeSampling
import time
from utils import save_training_state

VEC_DIM = 100
LR = 0.01
EPOCHS = 20
BATCH_SIZE = 16
PRINT_STEP = 100


def train(data_file_name,
          context_size,
          num_noise_words,
          vec_dim,
          vec_combine_method="sum",  # 这个地方是有加和和拼接两种方法的，这里我只实现了加和
          save_all = False,  # 是否保存每个epoch的参数
          generate_plot=True,
          ):
    dataset = load_data(data_file_name)
    dataset = MyDataset(dataset, context_size, num_noise_words)

    train_dataloader_params = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 1
    }

    train_loader = DataLoader(dataset, **train_dataloader_params)
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    # device = "cpu"
    vocab_size = len(dataset.vocabs)
    doc_size = len(dataset.docs)
    model = DM(VEC_DIM, doc_size, vocab_size)
    model.to(device)

    loss_function = NegativeSampling()

    optimizer = Adam(params=model.parameters(), lr=LR)

    best_loss = float("inf")

    prev_model_file_path = None
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        losses = []

        for step, data in enumerate(train_loader, 0):
            model.train()
            optimizer.zero_grad()

            middle, preamble, epilogue, doc_index, noise = \
                data["middle"], data["preamble"], data["epilogue"], data["doc_index"], data["noise"]

            middle, preamble, epilogue, doc_index, noise = \
                middle.to(device), preamble.to(device), epilogue.to(device), doc_index.to(device), noise.to(device)

            context_ids = torch.concat((preamble, epilogue), dim=1)
            output = model(context_ids, doc_index, noise)

            # print("output = ",output)
            loss = loss_function(output)
            losses.append(loss.item())


            model.zero_grad()

            if step % PRINT_STEP == 0:
                print("epoch: {} , step: {} ,loss: {}".format(epoch, step,loss.item()))

            loss.backward()
            optimizer.step()


        loss_mean = torch.mean(torch.FloatTensor(losses))
        is_best_loss = loss_mean < best_loss
        best_loss = min(loss_mean, best_loss)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_loss": best_loss,
            "optimizer_state_dict": optimizer
        }

        prev_model_file_path = save_training_state(
            data_file_name,
            vec_combine_method,
            context_size,
            num_noise_words,
            vec_dim,
            BATCH_SIZE,
            LR,
            epoch,
            loss_mean,
            state,
            save_all,
            generate_plot,
            is_best_loss,
            prev_model_file_path
        )

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss_mean))


if __name__ == "__main__":
    train(data_file_name="zh.txt",
          context_size=2,
          num_noise_words=2,
          vec_dim=100
          )