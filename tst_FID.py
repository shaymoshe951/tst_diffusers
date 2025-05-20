import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.linalg import sqrtm


def get_inception_features(images, batch_size=32, device='cuda'):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    model.fc = torch.nn.Identity()  # Remove final classifier

    features = []
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = torch.stack([preprocess(img) for img in batch]).to(device)
            feats = model(batch)
            features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)


def calculate_fid(real_feats, fake_feats):
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
