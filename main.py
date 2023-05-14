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
    # Prepare the data
    train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size)

    # Create the model
    model = ConvNet().to(device)
    print(f"Total parameters: {count_parameters(model)}")

    # Set up the criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    # Load the saved model (if it exists) and set the start_epoch
    start_epoch = 0
    if os.path.exists(saved_model_path):
        start_epoch = load_model(model, os.path.join(saved_model_path, "model_epoch.pt"), optimizer)
        print(f"Resuming training from epoch {start_epoch}")
    # Train and validate the model
    train_losses = []
    val_losses = []
    best_val_acc = 0
    with open("best_model/loss.csv", "w") as f: 
        f.write("train_loss,val_loss\n")
    for epoch in range(start_epoch,num_epochs):
        adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_period)
        train_loss, val_loss, val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epoch,num_epochs)
        # if epoch % 1 == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.2%}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        with open("best_model/loss.csv", "a") as f: 
            f.write(f"{train_loss},{val_loss}\n")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, saved_model_path, epoch, optimizer)

    # Plot the losses
    df = pd.read_csv("best_model/loss.csv")
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.legend()
    plt.savefig("best_model/losses.png")

    # Load the best model and evaluate on the test set
    best_epoch = load_model(model, "best_model/model_epoch{best_epoch}.pt", optimizer)
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()