# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from models import ConvNet
from train import train_and_validate, evaluate
from utils import prepare_data, plot_losses, count_parameters, save_model, load_model, adjust_learning_rate
import pandas as pd
import os

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device用的是:',device)
    print('现在的目录',os.getcwd())
    data_dir = "data/tiny-imagenet-200"
    batch_size = 20
    num_epochs = 100
    lr_decay_period = 30
    init_lr = 0.001
    saved_model_path = "best_model"

    # 准备数据
    train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size)

    # 创建模型
    model = ConvNet().to(device)
    print(f"总参数数量: {count_parameters(model)}")

    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # 加载已保存的模型（如果存在），并设置开始的epoch
    start_epoch = 0
    if os.path.exists(saved_model_path):
        start_epoch = load_model(model, os.path.join(saved_model_path, "model_epoch.pt"), optimizer)
        print(f"从第{start_epoch}个epoch开始继续训练")

    # 训练和验证模型
    train_losses = []
    val_losses = []
    best_val_acc = 0
    with open("best_model/loss.csv", "w") as f: 
        f.write("train_loss,val_loss\n")
    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_period)
        train_loss, val_loss, val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epoch, num_epochs)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        with open("best_model/loss.csv", "a") as f: 
            f.write(f"{train_loss},{val_loss}\n")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, saved_model_path, epoch, optimizer)

    # 绘制损失曲线
    df = pd.read_csv("best_model/loss.csv")
    plt.plot(df["train_loss"], label="训练损失")
    plt.plot(df["val_loss"], label="验证损失")
    plt.legend()
    plt.savefig("best_model/losses.png")

    # 加载最佳模型并在测试集上评估
    best_epoch = load_model(model, "best_model/model_epoch{best_epoch}.pt", optimizer)
    test_acc = evaluate(model, test_loader, device)
    print(f"测试准确率: {test_acc:.2%}")


if __name__ == "__main__":
    main()
