import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import ResNet, ResidualBlock
from data import get_datasets, get_dataloaders
from utils.config import config

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-tr", "--trainpath", help="Train data path")
parser.add_argument("-te", "--testpath", help="Test data path")

args = parser.parse_args()
train_data_path, val_data_path = args.trainpath, args.testpath

print(f"Train data path: {train_data_path}, test data path: {val_data_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is: {device}.")

img_transforms = transforms.Compose(
    [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()]
)

train_dataset, test_dataset = get_datasets(
    train_data_path, val_data_path, img_transforms
)
train_loader, test_loader = get_dataloaders(
    train_dataset, test_dataset, batch_size=config.BATCH_SIZE, shuffle=True
)

print(len(train_loader))
print(len(test_loader))

images, label = next(iter(train_loader))
print(images.shape)
print(label.shape)

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    momentum=config.MOMENTUM,
)
loss_history = []

for epoch in range(config.N_EPOCHS):
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
        print(f"Epoch {epoch+1} / {config.N_EPOCHS}. Accuracy: {100*correct/total} %.")

plt.plot(loss_history)
plt.show()
