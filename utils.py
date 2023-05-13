# utils.py
import torch
import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path, epoch, optimizer):
    """
    Save the model to the specified path, including epoch and optimizer state.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(path, f"model_epoch.pt"))

def load_model(model, path, optimizer=None):
    """
    Load the model from the specified path, including epoch and optimizer state (if provided).
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']

def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_period):
    """
    Adjust the learning rate of an optimizer based on the current epoch and decay period.
    """
    lr = init_lr * (0.1 ** (epoch // lr_decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def prepare_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform)

    # train_size = int(len(train_val_dataset) * 0.9)
    # val_size = len(train_val_dataset) - train_size

    # train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader



def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot the training and validation losses over epochs.

    Args:
        train_losses (list): A list of training losses.
        val_losses (list): A list of validation losses.
        save_path (str, optional): The path to save the plot. Defaults to None.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()