import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.sequencer import Shuffle
from torch import optim
from tqdm import tqdm
import logging

from noise_image_est import NoiseImageEst, NoiseImageEstImg2
from dataset_handler import get_mnist_data

# from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from scheduler import Scheduler, SchedulerLinear


def save_model(model, optimizer, epoch, cur_loss_vector, learning_rate, batch_size):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,  # Additional information if needed
        'loss': cur_loss_vector,
        'lr' : learning_rate,
        'batch_size' : batch_size
    }
    torch.save(checkpoint, 'model_checkpoint.pth')

def train(model : nn.modules, scheduler, nepochs, batch_size,
          learning_rate, check_point_fn = None, device='cuda',
          log_every = 100):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = get_mnist_data(batch_size)
    criterion = nn.MSELoss()

    cur_loss_vector = []
    starting_epoch = 0
    if check_point_fn:
        print(f"Loading checkpoing {check_point_fn}")
        checkpoint = torch.load(check_point_fn)
        print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        cur_loss_vector = checkpoint['loss']

    model.to('cuda')
    model.train()
    print("Starting training")
    for epoch in tqdm(range(starting_epoch, starting_epoch+nepochs)):
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

        running_loss_scaled = running_loss / len(train_loader)
        cur_loss_vector.append(running_loss_scaled)
        if epoch % log_every == 0 and epoch > 0:
            save_model(model,optimizer,epoch, cur_loss_vector, learning_rate, batch_size)
            print(f'Epoch {epoch + 1}, Loss: {running_loss_scaled}')

    plt.plot(cur_loss_vector)
    plt.show()
    print("Done")

# eval
def eval(model, scheduler, check_point_fn = None, device='cuda'):
    if check_point_fn:
        print(f"Loading checkpoing {check_point_fn}")
        checkpoint = torch.load(check_point_fn)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    xt = torch.randn((16,1,28,28),device=device)
    for t in reversed(scheduler.timesteps[1:]):
        t_t = torch.tensor([t],device=device).repeat(xt.shape[0])
        eps_th_est = model(xt,t_t)
        xt = scheduler.step_back(xt, eps_th_est, t)

    def show_images(images):
        fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
        for img, ax in zip(images, axes):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(f'Label')
            ax.axis('off')
        plt.show()

    show_images(xt.cpu().detach() * 2 + 1)



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    nepochs = 400  # 825
    batch_size = 32
    learning_rate = 0.001
    log_every = 20

    scheduler = SchedulerLinear() #Scheduler()
    model = NoiseImageEst(scheduler.T, (28, 28))
    # model = NoiseImageEstImg2(scheduler.T, (28, 28))
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model has {nparams} parameters")
    # train(model, scheduler, nepochs, batch_size, learning_rate,log_every=log_every)
    eval(model,scheduler, check_point_fn='model_checkpoint.pth')
    # eval(model,scheduler)