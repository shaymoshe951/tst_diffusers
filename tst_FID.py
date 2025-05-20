import torch
import numpy as np
# from torchvision.models import inception_v3
import torchvision.models as models
from torchvision import transforms
from scipy.linalg import sqrtm
from dataset_handler import get_mnist_data
from tqdm import tqdm


def get_inception_features(images, batch_size=32, device='cuda'):
    # model = inception_v3(pretrained=True, transform_input=False).to(device)
    # model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1').to(device)
    model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT').to(device)
    model.eval()
    model.fc = torch.nn.Identity()  # Remove final classifier

    features = []
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # preprocess = transforms.Compose([
    #     # transforms.Resize((299, 299)),  # Resize to 299x299
    #     transforms.Lambda(
    #         lambda x: x.repeat(3, 1, 1)
    #         if x.size(0) == 1 else x),  # Repeat channel if grayscale
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Inception V3 normalization
    # ])

    with torch.no_grad():
        if isinstance(images, torch.Tensor):
            # For images array
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch = torch.stack([preprocess(img) for img in batch]).to(device)
                feats = model(batch)
                features.append(feats.cpu().numpy())
        else:
            # For DataLoader
            for batch, _ in tqdm(images):
                batch = preprocess(batch.repeat(1,3,1,1)).to(device) # torch.stack([preprocess(img.squeeze()) for img in batch]).to(device)
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

if __name__ == "__main__":

    # Load MNIST data
    train, test = get_mnist_data(batch_size=8, flag_test=True)
    fid_calc = calculate_fid(get_inception_features(train), get_inception_features(test))
    print(f"FID: {fid_calc}, len(train): {len(train)}, len(test): {len(test)}")