#main_train.py
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        if self.split == 'train' or self.split == 'val':
            label_dirs = os.listdir(os.path.join(self.root_dir, self.split))
            for label_idx, label_dir in enumerate(label_dirs):
                image_dir = os.path.join(self.root_dir, self.split, label_dir, 'images')
                image_names = os.listdir(image_dir)

                for image_name in image_names:
                    self.images.append(os.path.join(image_dir, image_name))
                    self.labels.append(label_idx)

        else:
            with open(os.path.join(self.root_dir, self.split, 'val_annotations.txt')) as f:
                for line in f.readlines():
                    image_name, label_idx = line.split('\t')[:2]
                    self.images.append(os.path.join(self.root_dir, self.split, 'images', image_name))
                    self.labels.append(int(label_idx))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Hyperparameters and settings
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loader and transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    ])

    tiny_imagenet_train = TinyImageNetDataset(root_dir='data/tiny-imagenet-200', split='train', transform=transform)
    train_loader = DataLoader(tiny_imagenet_train, batch_size=batch_size, shuffle=True, num_workers=4)

    tiny_imagenet_val = TinyImageNetDataset(root_dir='data/tiny-imagenet-200', split='val', transform=transform)
    val_loader = DataLoader(tiny_imagenet_val, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, loss, and optimizer
    model = ConvNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
if __name__ == '__main__':
    main()