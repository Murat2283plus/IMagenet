import torch
import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

def count_parameters(model):
    """
    计算模型中可训练参数的总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path, epoch, optimizer):
    """
    将模型保存到指定路径，包括当前的训练轮数和优化器状态。
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
    从指定路径加载模型，包括训练轮数和优化器状态（如果提供了优化器）。
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']

def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_period):
    """
    根据当前训练轮数和学习率衰减周期调整优化器的学习率。
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
    绘制训练和验证损失随训练轮数变化的曲线图。

    参数:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        save_path (str, 可选): 保存图像的路径。默认为 None，显示图像。
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, val_losses, label='验证损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
