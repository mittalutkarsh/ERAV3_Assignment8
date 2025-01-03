from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Model_9 import Net
from data import get_dataloaders
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# For reproducibility
torch.manual_seed(1)

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

if cuda:
    torch.cuda.manual_seed(1)

# dataloader arguments - simple version with single worker
dataloader_args = dict(
    shuffle=True,
    batch_size=128,
    num_workers=0,  # Changed back to 0
    pin_memory=False
)

# Calculate dataset mean (you should calculate this from your dataset)
# DATASET_MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR10 mean
# DATASET_STD = (0.2470, 0.2435, 0.2616)   # CIFAR10 std

# # Define Albumentations transformations
# train_transforms = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
#     A.CoarseDropout(
#         max_holes=1,        # Reduced from 3 to 1
#         max_height=16,      # Increased to 16
#         max_width=16,       # Increased to 16
#         min_holes=1,
#         min_height=4,       # Fixed value
#         min_width=4,        # Fixed value
#         fill_value=0,
#         p=0.5
#     ),
#     A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
#     ToTensorV2()
# ])

# test_transforms = A.Compose([
#     A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
#     ToTensorV2()
# ])

# Get train dataloader and train data
data_dir = './data'
train_loader, test_loader, exp_data_train, exp_test = get_dataloaders(
    data_dir, 
    dataloader_args['batch_size'], 
    dataloader_args['num_workers'], 
    dataloader_args['pin_memory']
)[:4]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

model = Net().to(device)

# Calculate and display total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal Parameters: {total_params:,}')
print(f'Trainable Parameters: {trainable_params:,}\n')

# Add these debug lines
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print(f"{name}: {param.shape}")

summary(model, input_size=(3, 32, 32))

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

EPOCHS = 16
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

    print(f'Train: Average Accuracy={100 * correct / processed:0.2f}%')
    train_acc.append(100 * correct / processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    acc = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n'
    )

    test_acc.append(acc)

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader)

    # Print train and test accuracy
    print("\nTrain Accuracy:", train_acc[-1])
    print("Test Accuracy:", test_acc[-1]) 