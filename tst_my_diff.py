import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.sequencer import Shuffle
from torch import optim
from tqdm import tqdm
import logging

from noise_image_est import NoiseImageEst
from dataset_handler import get_mnist_data

# from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

class Scheduler():
    def __init__(self, device='cuda'):
        self.device = device
        self.T = 1000
        self.timesteps = torch.arange(0,self.T).to(device)
        t = self.timesteps
        T = self.T
        s = 0.008
        self.s = s
        # calculate ft
        ftv = torch.cos( (t/T + s)/(1+s) * torch.pi/2).to(device)
        self.alphatv = ftv/ftv[0]
        # plt.plot(t,alphat)
        # plt.show()

    def sample(self, x0):
        t = torch.randint(0, self.T - 1, (x0.shape[0],), device = self.device)
        t = t.view(t.shape[0],1,1,1)
        alphat = self.alphatv[t]
        eps_xt_t = torch.randn(x0.shape, device=self.device)
        xt = torch.sqrt(alphat) * x0 + torch.sqrt(1-alphat) * eps_xt_t
        return xt, eps_xt_t, t



# sc = Scheduler()
# y = sc.get_step(torch.zeros((4,5)),10)


def train(model : nn.modules, scheduler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_mnist_data(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Set the model in training mode
    model.to('cuda')
    model.train()
    for epoch in tqdm(range(nepochs)):
        iter = 0
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the grad
            optimizer.zero_grad()

            # Prepare the input to the model
            x0 = images.to(device, non_blocking=True)
            xt, eps_xt_t, t = scheduler.sample(x0)
            eps_th_xt_t = model(xt, t)

            loss = criterion(eps_th_xt_t, eps_xt_t) # TODO: Tmp
            loss.backward()
            optimizer.step()
            iter += 1
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    nepochs = 5  # 825
    batch_size = 32
    learning_rate = 0.001

    scheduler = Scheduler()
    model = NoiseImageEst(scheduler.T, (28, 28))

    train(model, scheduler)