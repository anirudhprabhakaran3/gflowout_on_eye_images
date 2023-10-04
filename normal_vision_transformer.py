import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
from models.VisionTransformer import VisionTransformer
from data import get_datasets, get_dataloaders
from tqdm import tqdm
import gc
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-tr", "--trainpath", help="Train data path", required=True, type=str
)
parser.add_argument("-te", "--testpath", help="Test data path", required=True, type=str)
parser.add_argument("-size", "--img-size", help="Image size", default=224, type=int)
parser.add_argument("-b", "--batch-size", help="Batch size", default=32, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=10, type=int)
parser.add_argument(
    "-lr", "--learning-rate", help="Learning Rate", default=0.001, type=float
)
parser.add_argument(
    "-d", "--depth", help="Depth (no. of attention blocks)", default=12, type=int
)
parser.add_argument(
    "-n", "--n-heads", help="Number of atttention heads", default=12, type=int
)
parser.add_argument("-mlp", "--mlp-ratio", help="MLP Ratio", default=4.0, type=float)
parser.add_argument("-bias", "--qkv-bias", help="QKV Bias", default=True, type=float)
parser.add_argument(
    "-p", "--prob", help="Droput probability for network", default=0.0, type=float
)
parser.add_argument(
    "-attnp",
    "--attn-p",
    help="Dropout probability for attention module",
    default=0.0,
    type=float,
)

args = parser.parse_args()

print("Arguments provided are:")
print(args)

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

images, label = next(iter(train_loader))
images, label = images.to(device), label.to(device)
print(f"Image shape: {images.shape}")
print(f"Label shape: {label.shape}")

model = VisionTransformer(
    img_size=args.img_size,
    depth=args.depth,
    n_heads=args.n_heads,
    mlp_ratio=args.mlp_ratio,
    qkv_bias=args.qkv_bias,
    p=args.prob,
    attn_p=args.attn_p,
)

summary(model, input_data=images)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_history = []

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}")

    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss_history.append(loss)

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
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs, predicted
        print(f"Accuracy: {100*correct/total}")
