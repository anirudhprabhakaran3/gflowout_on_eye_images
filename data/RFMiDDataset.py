import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image

class RFMIDImageDataset(Dataset):
    def __init__(self, images_path, annotations_path, preprocess):
        super().__init__()

        self.images_path = images_path
        self.annotations_path = annotations_path
        self.preprocess = preprocess

        self.annotations = pd.read_csv(self.annotations_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = self.images_path + "/" + str(row["ID"]) + ".png"
        img = Image.open(image_path)
        img = self.preprocess(img)

        labels = row[2:].astype('float32')
        labels = torch.argmax(torch.tensor(labels))

        return img, labels