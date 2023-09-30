import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import ViT
from data import get_datasets, get_dataloaders

IMG_SIZE = 224
BATCH_SIZE = 32
N_CLASSES = 2
N_EPOCHS = 10
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is: {device}.")

train_data_path = "/home/anirudh/mldata/ocular-disease/data/experiments/train"
val_data_path = "/home/anirudh/mldata/ocular-disease/data/experiments/val"

img_transforms = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

train_dataset, test_dataset = get_datasets(
    train_data_path, val_data_path, img_transforms
)
train_loader, test_loader = get_dataloaders(
    train_dataset, test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

print(len(train_loader))
print(len(test_loader))

images, label = next(iter(train_loader))
print(images.shape)
print(label.shape)

model = ViT(patch_size=32, emb_size=100, depth=1).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM
)
loss_history = []

for epoch in range(N_EPOCHS):
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
        print(f"Epoch {epoch+1} / {N_EPOCHS}. Accuracy: {100*correct/total} %.")

plt.plot(loss_history)
plt.show()
