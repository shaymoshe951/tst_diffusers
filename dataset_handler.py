from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory=True, num_workers = 4, persistent_workers=True)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,  pin_memory=True
    #                 num_workers = 0, persistent_workers = True)

    return train_loader

if __name__=="__main__":
    import matplotlib.pyplot as plt

    train_loader = get_mnist_data(16)
    # Function to visualize images
    def show_images(images, labels):
        fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
        for img, label, ax in zip(images, labels, axes):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        plt.show()

    # Get a batch of images from the training data loader
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Display the first 5 images
    show_images(images[:5], labels[:5])