import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from Model_9 import Net

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np

# Import your model here:
# e.g., from Model_9 import Net

# --------------------------
# 1) Albumentations Transforms
#    - Example: gentler rotation, no CoarseDropout initially
# --------------------------
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, 
        scale_limit=0.1, 
        rotate_limit=15, 
        p=0.5
    ),
    A.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2470, 0.2435, 0.2616),
        p=1.0
    ),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2470, 0.2435, 0.2616),
        p=1.0
    ),
    ToTensorV2()
])

# --------------------------
# 2) Custom CIFAR-10 Dataset using Albumentations
# --------------------------
class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.albumentations_transform is not None:
            transformed = self.albumentations_transform(image=image)
            image = transformed["image"]  # Albumentations -> Tensor
        return image, label

    def __len__(self):
        return len(self.data)

# --------------------------
# 3) Setup Data, Device, & Loaders
# --------------------------
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

trainset = Cifar10SearchDataset(
    root='./data',
    train=True,
    download=True,
    transform=train_transforms
)
testset = Cifar10SearchDataset(
    root='./data',
    train=False,
    download=True,
    transform=test_transforms
)

dataloader_args = dict(
    batch_size=64,     # Larger batch size
    shuffle=True,
    num_workers=2,     # Parallel data loading
    pin_memory=True    # Speed up data transfer to GPU (if supported)
)

train_loader = DataLoader(trainset, **dataloader_args)
test_loader = DataLoader(
    testset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True
)

# --------------------------
# 4) Instantiate Your Model
#    Make sure it returns log_softmax if using F.nll_loss
# --------------------------
# from Model_9 import Net
model = Net().to(device)

print("\nModel Summary:")
summary(model, input_size=(3, 32, 32))

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Parameters: {total_params:,}')
print(f'Trainable Parameters: {trainable_params:,}\n')

# --------------------------
# 5) Optimizer & Scheduler
#    - Gentler StepLR instead of big exponential decay
# --------------------------
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

EPOCHS = 50
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# --------------------------
# 6) Training & Testing Functions
# --------------------------
def train(model, device, train_loader, optimizer):
    model.train()
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # log_softmax output expected if using nll_loss
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

    avg_train_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / processed
    train_losses.append(avg_train_loss)
    train_acc.append(accuracy)
    print(f"Train: Average Loss={avg_train_loss:.4f}, Accuracy={accuracy:.2f}%")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(accuracy)
    print(f"Test : Average Loss={test_loss:.4f}, Accuracy={accuracy:.2f}%\n")

# --------------------------
# 7) Main Training Loop
# --------------------------
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        print(f"EPOCH: {epoch}")
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    print("Final Train Accuracy:", train_acc[-1])
    print("Final Test Accuracy:", test_acc[-1])
