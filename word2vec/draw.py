import matplotlib.pyplot as plt
import re

def draw_loss(file_path):
    plt.figure(figsize=(15,7.5),dpi=300)
    with open(file_path,"r") as f:
        content = f.read()
        contents = content.split("\n")

    losses = []
    steps = []

    loss_pattern = re.compile(r'\d\.\d+')
    # epoch_pattern = re.compile(r'epoch: \d+ ')
    # step_pattern = re.compile(r', step: \d+ ')
    for content in contents:
        if re.match('epoch: \d+ , step:',content) != None:
            loss = float(loss_pattern.findall(content)[0])
            if loss < 3:
                losses.append(loss)

    for i in range(len(losses)):
        steps.append(i*100)
    # print("losses = ",losses)
    # print("steps = ",steps)
    plt.plot(steps,losses)
    plt.show()


if __name__ == "__main__":
    draw_loss('log')


