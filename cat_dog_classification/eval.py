import torch
import numpy as np
from torch import cuda
from sklearn.metrics import classification_report
import torch.nn.functional as F
import torch.nn as nn

def eval(model,save_model,testing_loader,writer,epoch,one_epoch_size,step):
    device = 'cuda' if cuda.is_available() else 'cpu'

    if save_model != None:
        model.load_state_dict(torch.load(save_model))
    model.to(device)
    model.eval()

    the_label = []
    the_prediction = []
    losses = []

    BCE =nn.BCELoss()

    with torch.no_grad():
        for step, data in enumerate(testing_loader, 0):
            img, label = data["img"], data["label"]
            img_device = img.to(device)
            output = model(img_device)
            one_hot_label = F.one_hot(label,num_classes=2).to(device,dtype=torch.float)
            loss = BCE(output,one_hot_label)
            output = torch.argmax(output,axis=1).cpu()
            losses.append(loss)
            # print(output)
            label = label.cpu()
            the_label.append(label)
            the_prediction.append(output)

        # print("the_label = ",the_label)
        # print("the_prediction = ",the_prediction)
        assert len(the_label) == len(the_prediction)
        # if len(the_label) > 1:
        the_label,the_prediction = np.concatenate(the_label),np.concatenate(the_prediction)
        if len(losses) > 1:
            loss = np.concatenate(losses)
        else:
            loss = losses[0]
        # print("eval/loss = ",loss)

        result = classification_report(the_label,the_prediction)
        writer.add_scalar(tag="loss/eval",scalar_value=loss.item(),global_step=epoch*one_epoch_size+step)
        # print("result['accuracy'] = ",result[0])
        writer.add_scalar(tag="acc/eval",scalar_value=classification_report(the_label,the_prediction,output_dict=True)["accuracy"],global_step=epoch*one_epoch_size+step)
        print("result = ",result)


