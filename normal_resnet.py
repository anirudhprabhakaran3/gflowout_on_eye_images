import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import ResNet, ResidualBlock
from data import get_datasets, get_dataloaders
from utils.options import options

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-tr", "--trainpath", help="Train data path", required=True, type=str
)
parser.add_argument("-te", "--testpath", help="Test data path", required=True, type=str)
parser.add_argument("-i", "--img-size", help="Image Size", default=224, type=int)
parser.add_argument("-b", "--batch-size", help="Batch Size", default=32, type=int)
parser.add_argument("-e", "--epochs", help="Epochs", default=10, type=int)
parser.add_argument(
    "-lr", "--learning-rate", help="Learning Rate", default=0.001, type=float
)
parser.add_argument("-m", "--momentum", help="Momentum", default=0.9, type=float)
parser.add_argument(
    "-wd", "--weight-decay", help="Weight Decay", default=0.1, type=float
)

args = parser.parse_args()
train_data_path, val_data_path = args.trainpath, args.testpath

print(f"Train data path: {train_data_path}, test data path: {val_data_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is: {device}.")

img_transforms = transforms.Compose(
    [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
)

train_dataset, test_dataset = get_datasets(
    train_data_path, val_data_path, img_transforms
)
train_loader, test_loader = get_dataloaders(
    train_dataset, test_dataset, batch_size=args.batch_size, shuffle=True
)

print(len(train_loader))
print(len(test_loader))

images, label = next(iter(train_loader))
images, label = images.to(device), label.to(device)
print(f"Image shape: {images.shape}")
print(f"Label shape: {label.shape}")

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
summary(model, input_data=images)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    momentum=args.momentum,
)
loss_history = []

for epoch in range(args.epochs):
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        print(f"Epoch {epoch+1} / {args.epochs}. Accuracy: {100*correct/total} %.")

plt.plot(loss_history)
plt.show()
