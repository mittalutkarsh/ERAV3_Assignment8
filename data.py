import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


SEED = 1


class AlbumentationsDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)  # Convert PIL image to numpy array
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data_dir, batch_size, num_workers, pin_memory):
    # CIFAR10 expects 32x32 color images, so we'll use appropriate transforms
    simple_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    exp_train = datasets.CIFAR10(
        data_dir, 
        train=True, 
        download=True, 
        transform=simple_transforms
    )
    exp_test = datasets.CIFAR10(
        data_dir, 
        train=False, 
        download=True, 
        transform=test_transforms
    )

    exp_data_train = exp_train.data

    # Calculate dataset mean (you should calculate this from your dataset)
    DATASET_MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR10 mean
    DATASET_STD = (0.2470, 0.2435, 0.2616)   # CIFAR10 std

    # Define Albumentations transformations
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_height=8,
            max_width=8,
            min_holes=1,
            min_height=1,
            min_width=1,
            fill_value= sum(DATASET_MEAN)/3,
            p=0.5
        ),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2()
    ])

    # Wrap datasets with Albumentations
    train_dataset = AlbumentationsDataset(exp_train, train_transforms)
    test_dataset = AlbumentationsDataset(exp_test, test_transforms)

    dataloader_args = dict(
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    train_loader, test_loader, train_data, test_data, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, exp_data_train, exp_test, test_loader

if __name__ == "__main__":
    # Test configuration
    data_dir = './data'  # Data will be downloaded here
    batch_size = 64
    num_workers = 2
    pin_memory = True

    # Get the dataloaders
    train_loader, test_loader, train_data, test_data, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Print some basic information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Number of classes: {len(train_loader.dataset.classes)}")
    print(f"Classes: {train_loader.dataset.classes}")
