import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image

class ODIRImageDataset(Dataset):
    def __init__(self, images_path, annotations_path, preprocess):
        super().__init__()

        self.images_path = images_path
        self.annotations_path = annotations_path
        self.annotations = pd.read_csv(self.annotations_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        image_path = self.images_path + "/" + row["Fundus"]
        img = Image.open(image_path)
        img = self.preprocess(img)
        labels = row[2:-1].astype('float32')
        labels = torch.tensor(labels)
        labels = torch.argmax(labels)

        return img, labels