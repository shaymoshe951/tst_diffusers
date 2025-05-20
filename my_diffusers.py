import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.sequencer import Shuffle
from torch import optim
from tqdm import tqdm
import logging

from unets_imp import UNetS
from dataset_handler import get_mnist_data
from utils import save_images, setup_logging


class Diffusers:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, nchannels = 3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, nchannels, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    setup_logging(args.run_name)
    # Initialize the model
    model = UNetS(
        image_shape_hwc=(args.img_size, args.img_size, 1),
        device=args.device
    ).to(args.device)

    # Initialize the diffusion process
    diffusion = Diffusers(
        noise_steps=args.noise_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        img_size=args.img_size,
        device=args.device
    )

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    dataset = get_mnist_data(args.batch_size)
    l = len(dataset)
    model.train()
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataset)
        for i, (images, _) in enumerate(pbar):
            images = images.to(args.device)
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            xt, eps_xt_t = diffusion.noise_images(images, t)

            optimizer.zero_grad()
            eps_th_xt_t = model(xt, t)
            loss = mse(eps_th_xt_t, eps_xt_t)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item()}")

        sampled_images = diffusion.sample(model, n=images.shape[0], nchannels=1)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


if __name__ == "__main__":
    # Example usage
    class Args:
        img_size = 28
        noise_steps = 1000
        beta_start = 1e-4
        beta_end = 0.02
        batch_size = 32
        epochs = 10
        lr = 2e-4
        run_name = "DDPM_Uncondtional1"
        dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
        device = "cuda"
        mode = 'train' # 'train' or 'eval'

    args = Args()
    if args.mode == 'eval':
        # Load the model and evaluate
        model = UNetS((args.img_size, args.img_size, 1)).to(args.device)
        model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt")))
        diffusion = Diffusers(
            noise_steps=args.noise_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            img_size=args.img_size,
            device=args.device
        )
        sampled_images = diffusion.sample(model, n=8, nchannels=1)
        save_images(sampled_images, os.path.join("results", args.run_name, "sampled.jpg"))
    else:
        train(args)