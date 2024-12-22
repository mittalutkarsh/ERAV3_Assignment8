import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


SEED = 1


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

    dataloader_args = dict(
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    train_loader = torch.utils.data.DataLoader(exp_train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(exp_test, **dataloader_args)

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
