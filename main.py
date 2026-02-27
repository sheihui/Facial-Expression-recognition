import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # to fix "fatal error: KMP duplicate lib" on macOS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from fer import Fer2013
from models.vgg import VGG19
from models.resnet import ResNet18
from utils import clip_gradient

batch_size = 128
lr = 0.01
num_epochs = 250
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
grad_clip = 0.5


# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.CenterCrop(44),
    transforms.ToTensor(),
])


# load data
train_set = Fer2013(split="train", transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = Fer2013(split="test", transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# init the models
model = ResNet18(num_classes=num_classes).to(device)
# model = VGG19(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
shceduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # clip gradients to prevent exploding gradients
        clip_gradient(optimizer, grad_clip)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def test_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# training loop
best_acc = 0

for epoch in range(num_epochs):
    train_loss, tarin_acc = train_epoch(train_loader, model, criterion, optimizer, device)
    test_loss, test_acc = test_epoch(test_loader, model, criterion, device)

    shceduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {tarin_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%", flush=True)
    print("-" * 50, flush=True)


save_path = "/root/autodl-tmp/.autodl-tmp/FEC/weights/resnet_original.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"✅ 训练完成！模型已保存到：{save_path}")




print(f"Train Loss: {train_loss:.4f}, Train Acc: {tarin_acc:.2f}% | "
    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%", flush=True)
