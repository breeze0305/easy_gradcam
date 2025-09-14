import os
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ConcreteCrackDataset

from model import Model

def main():
    data_dir = "./data"
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])


    train_dataset = ConcreteCrackDataset(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = ConcreteCrackDataset(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)


    # model = timm.create_model("resnet18", pretrained=True, num_classes=2)
    model = Model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print("開始訓練模型...")
    for epoch in range(num_epochs):
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} ")
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")


    os.makedirs("runs", exist_ok=True)

    # Loss 
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("runs/loss_curve.png")
    plt.close()

    # Accuracy 
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("runs/accuracy_curve.png")
    plt.close()

    torch.save(model.state_dict(), "runs/best.pth")
    print("訓練完成！")


if __name__ == "__main__":
    main()