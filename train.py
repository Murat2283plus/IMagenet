import torch

def train(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    for inputs, targets in train_loader:
        #inputs.shape #torch.Size([128, 3, 64, 64])
        #targets.shape #torch.Size([128])
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == targets).item()
            total_predictions += targets.size(0)

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions

    return epoch_loss, accuracy

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epoch,num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Accuracy: {val_acc:.2%}')

    return train_loss, val_loss, val_acc

def evaluate(model, test_loader, device):
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == targets).item()
            total_predictions += targets.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy