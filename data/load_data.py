from torchvision import datasets
from torch.utils.data import DataLoader


def get_datasets(train_path, test_path, img_transforms):
    train_dataset = datasets.ImageFolder(train_path, transform=img_transforms)
    test_dataset = datasets.ImageFolder(test_path, transform=img_transforms)

    return (train_dataset, test_dataset)


def get_dataloaders(train_dataset, test_dataset, batch_size=32, shuffle=True):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )

    return (train_loader, test_loader)
