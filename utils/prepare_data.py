# This file is prepared from Google Colab.

# !git clone https://github.com/PerceptiLabs/ocular-disease.git

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from torchvision import transforms
import os
from config import config

df = pd.read_csv("/content/ocular-disease/data/dataset.csv")
DATA_PATH = "/content/ocular-disease/data"
X = df["filename"]
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

TRAIN_DATALOADER_FOLDER = "/content/dataloader/train"
VAL_DATALOADER_FOLDER = "/content/dataloader/val"

os.makedirs(f"{TRAIN_DATALOADER_FOLDER}/C")
os.makedirs(f"{TRAIN_DATALOADER_FOLDER}/N")
os.makedirs(f"{VAL_DATALOADER_FOLDER}/C")
os.makedirs(f"{VAL_DATALOADER_FOLDER}/N")

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

img_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)


def create_files(dataset, dataloader_folder):
    for i, filepath in enumerate(dataset):
        label = y_train.iloc[i]
        file_name = filepath.split("/")[-1]
        original_path = f"{DATA_PATH}/{filepath}"
        new_path = f"{dataloader_folder}/{label}/{file_name}"
        shutil.copyfile(original_path, new_path)


create_files(X_train, TRAIN_DATALOADER_FOLDER)
create_files(X_test, VAL_DATALOADER_FOLDER)

# from torchvision import datasets
# from torch.utils.data import DataLoader

# train_dataset = datasets.ImageFolder(
#     "/content/dataloader/train", transform=img_transforms
# )
# val_dataset = datasets.ImageFolder("/content/dataloader/val", transform=img_transforms)

# train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
