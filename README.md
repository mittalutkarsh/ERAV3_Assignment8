# Utkarsh-ERA-V2-Assignment-7
Assignment 7
# MNIST Digit Classification using Convolutional Neural Networks (CNNs)

This project explores the use of Convolutional Neural Networks (CNNs) for digit classification on the MNIST dataset. We experiment with different model architectures and hyperparameters to achieve high accuracy in classifying handwritten digits.

## Project Structure

The project is organized into three main modules:

1. **data**: This module handles the data loading and preprocessing. It includes the `data.py` file, which defines the `get_dataloaders` function. This function loads the MNIST dataset, applies transformations, and returns the data loaders for training and testing.

2. **models**: This module contains the implementation of different CNN architectures. Each model is defined in a separate file (e.g., `Model_1.py`, `Model_2.py`, etc.) and follows a similar structure. The models are implemented using the PyTorch framework and vary in terms of the number and size of convolutional layers, pooling layers, and fully connected layers.

3. **main**: The main module consists of the `main.ipynb` file, which is a Jupyter Notebook that brings together the data and model modules. It handles the training and evaluation of the models. The notebook loads the data, defines the model architecture, sets up the optimizer and loss function, and performs the training and testing loops. It also includes code for visualizing the results and saving the trained models.

## Dataset

The MNIST dataset is a widely-used benchmark dataset for image classification tasks. It consists of 70,000 grayscale images of handwritten digits, with each image being 28x28 pixels in size. The dataset is split into 60,000 training images and 10,000 test images.

## Model Architectures

In this project, we experiment with 8 different CNN architectures to explore their performance on the MNIST dataset. The models are implemented using the PyTorch framework and vary in terms of the number and size of convolutional layers, pooling layers, and fully connected layers.

### Model 1

#### Target:

Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible.
No fancy stuff
#### Results:
Parameters: 6,379,786
Best Train Accuracy: 100.00
Best Test Accuracy: 99.21
- **No of EPOCHS used:** 20
#### Analysis:
The model is quite large, but it is working well.
We observe some slight overfitting, as the train accuracy reaches 100% while the test accuracy is slightly lower.

| Layer             | Kernel Size | Stride | Padding | Receptive Field |
|-------------------|-------------|--------|---------|-----------------|
| Input             | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)  | 3           | 1      | 1       | 3x3             |
| Layer 2 (Conv2d)  | 3           | 1      | 1       | 5x5             |
| Layer 3 (MaxPool2d)| 2         | 2      | 0       | 6x6             |
| Layer 4 (Conv2d)  | 3           | 1      | 1       | 10x10           |
| Layer 5 (Conv2d)  | 3           | 1      | 1       | 14x14           |
| Layer 6 (MaxPool2d)| 2         | 2      | 0       | 16x16           |
| Layer 7 (Conv2d)  | 3           | 1      | 0       | 24x24           |
| Layer 8 (Conv2d)  | 3           | 1      | 0       | 32x32           |
| Layer 9 (Conv2d)  | 3           | 1      | 0       | 40x40           |




### Model 2

#### Target:

Reduce the number of parameters while maintaining the model performance.
Introduce batch normalization and dropout for regularization.
#### Results:
Parameters: 10,970
Best Train Accuracy: 99.87
Best Test Accuracy: 99.20
- **No of EPOCHS used:** 20
#### Analysis:
The model architecture is optimized, resulting in a significant reduction in the number of parameters compared to Model 1.
The inclusion of batch normalization and dropout helps in regularizing the model and improving its generalization ability.
The model achieves high accuracy on both the training and test sets, indicating good performance and reduced overfitting.

| Layer              | Kernel Size | Stride | Padding | Receptive Field |
|--------------------|-------------|--------|---------|-----------------|
| Input              | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)   | 3           | 1      | 1       | 3x3             |
| Layer 2 (Conv2d)   | 3           | 1      | 1       | 5x5             |
| Layer 3 (Conv2d)   | 3           | 1      | 1       | 7x7             |
| Layer 4 (MaxPool2d)| 2           | 2      | 0       | 8x8             |
| Layer 5 (Conv2d)   | 1           | 1      | 0       | 8x8             |
| Layer 6 (Conv2d)   | 3           | 1      | 1       | 12x12           |
| Layer 7 (Conv2d)   | 3           | 1      | 1       | 16x16           |
| Layer 8 (MaxPool2d)| 2           | 2      | 0       | 18x18           |
| Layer 9 (Conv2d)   | 1           | 1      | 0       | 18x18           |
| Layer 10 (Conv2d)  | 7           | 1      | 0       | 36x36           |


### Model 3

[Provide a brief description of Model 2 architecture and its key features]

### Model 4

#### Target:

Keeping the same number of parameters while maintaining the model performance.
Introduce Global Average Pooling (GAP) and remove the last BIG kernel.
#### Results:
Parameters: 11,312
Best Train Accuracy: 99.28
Best Test Accuracy: 99.30
- **No of EPOCHS used:** 20
#### Analysis:
The model architecture has been modified with the introduction of Global Average Pooling (GAP) and the removal of the last BIG kernel.
Despite the changes, the number of parameters has slightly increased compared to Model 2 (10,970 parameters). This increase can be attributed to the additional convolutional layers and the increased number of channels in some layers.
However, the model still maintains high accuracy on both the training and test sets, indicating good performance.
The use of GAP helps in reducing the spatial dimensions of the feature maps while retaining the important features, providing a more robust feature representation.

| Layer             | Kernel Size | Stride | Padding | Receptive Field |
|-------------------|-------------|--------|---------|-----------------|
| Input             | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)  | 3           | 1      | 0       | 3x3             |
| Layer 2 (Conv2d)  | 3           | 1      | 0       | 5x5             |
| Layer 3 (Conv2d)  | 1           | 1      | 0       | 5x5             |
| Layer 4 (MaxPool2d)| 2          | 2      | 0       | 6x6             |
| Layer 5 (Conv2d)  | 3           | 1      | 0       | 10x10           |
| Layer 6 (Conv2d)  | 3           | 1      | 0       | 14x14           |
| Layer 7 (Conv2d)  | 3           | 1      | 0       | 18x18           |
| Layer 8 (Conv2d)  | 3           | 1      | 1       | 20x20           |
| Layer 9 (AvgPool2d)| 6          | 1      | 0       | 32x32           |
| Layer 10 (Conv2d) | 1           | 1      | 0       | 32x32           |


### Model 5

#### Target

- Reduce the number of parameters while maintaining the model performance.
- Use a smaller number of channels in the initial convolutional layers.

#### Results

- **Parameters:** 9,400 
- **Best Train Accuracy:** 99.21%
- **Best Test Accuracy:** 99.40%
- **No of EPOCHS used:** 20

#### Analysis

The model architecture is optimized by reducing the number of channels in the initial convolutional layers, resulting in a significant reduction of parameters compared to the previous models. Despite the reduction in parameters, the model maintains high accuracy on both the training and test sets. The use of a smaller number of channels in the initial layers helps in reducing the computational complexity while still capturing the important features. The model demonstrates good generalization ability, as evidenced by the high test accuracy.

## Receptive Field Calculation

| Layer             | Kernel Size | Stride | Padding | Receptive Field |
|-------------------|-------------|--------|---------|-----------------|
| Input             | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)  | 3           | 1      | 0       | 3x3             |
| Layer 2 (Conv2d)  | 3           | 1      | 0       | 5x5             |
| Layer 3 (Conv2d)  | 1           | 1      | 0       | 5x5             |
| Layer 4 (MaxPool2d)| 2          | 2      | 0       | 6x6             |
| Layer 5 (Conv2d)  | 3           | 1      | 0       | 10x10           |
| Layer 6 (Conv2d)  | 3           | 1      | 0       | 14x14           |
| Layer 7 (Conv2d)  | 3           | 1      | 0       | 18x18           |
| Layer 8 (Conv2d)  | 3           | 1      | 1       | 20x20           |
| Layer 9 (AvgPool2d)| 6          | 1      | 0       | 32x32           |
| Layer 10 (Conv2d) | 1           | 1      | 0       | 32x32           |


### Model 6

#### Target

- Introduce image augmentation techniques to improve model generalization.
- Reduce the number of epochs to 15 while maintaining model performance.
- Further optimize the model architecture to reduce the number of parameters.

#### Results

- **Parameters:** 9,256
- **Best Train Accuracy:** 98.65%
- **Best Test Accuracy:** 99.06%
-  **No of EPOCHS used:** 15

#### Analysis

- The model incorporates image augmentation techniques, which helps in improving the model's ability to generalize well to unseen data.
- Despite reducing the number of epochs to 15, the model maintains high performance, indicating efficient learning and convergence.
- The model architecture is further optimized, resulting in a reduction of parameters compared to the previous models.
- The model achieves a good balance between training and test accuracy, suggesting reduced overfitting and improved generalization.

#### Receptive Field Calculation

| Layer             | Kernel Size | Stride | Padding | Receptive Field |
|-------------------|-------------|--------|---------|-----------------|
| Input             | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)  | 3           | 1      | 0       | 3x3             |
| Layer 2 (Conv2d)  | 3           | 1      | 0       | 5x5             |
| Layer 3 (Conv2d)  | 1           | 1      | 0       | 5x5             |
| Layer 4 (MaxPool2d)| 2          | 2      | 0       | 6x6             |
| Layer 5 (Conv2d)  | 3           | 1      | 0       | 10x10           |
| Layer 6 (Conv2d)  | 3           | 1      | 0       | 14x14           |
| Layer 7 (Conv2d)  | 3           | 1      | 0       | 18x18           |
| Layer 8 (Conv2d)  | 3           | 1      | 1       | 20x20           |
| Layer 9 (AvgPool2d)| 6          | 1      | 0       | 32x32           |
| Layer 10 (Conv2d) | 1           | 1      | 0       | 32x32           |


### Model 7

#### Target

- Optimize the model architecture for better performance.
- Incorporate adaptive average pooling for flexible input sizes.
- Utilize an adaptive learning rate scheduler for improved convergence.

#### Results

- **Parameters:** 7,568
- **Best Train Accuracy:** 98.59%
- **Best Test Accuracy:** 99.11%
-  **No of EPOCHS used:** 15

#### Analysis

- The model architecture is optimized, resulting in improved performance compared to the previous models.
- The incorporation of adaptive average pooling allows the model to handle input images of varying sizes, making it more versatile.
- The use of an adaptive learning rate scheduler helps in achieving better convergence during training.
- The model maintains a good balance between training and test accuracy, indicating effective generalization.

#### Receptive Field Calculation

| Layer                  | Kernel Size | Stride | Padding | Receptive Field |
|------------------------|-------------|--------|---------|-----------------|
| Input                  | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)       | 3           | 1      | 0       | 3x3             |
| Layer 2 (Conv2d)       | 3           | 1      | 0       | 5x5             |
| Layer 3 (Conv2d)       | 1           | 1      | 0       | 5x5             |
| Layer 4 (MaxPool2d)    | 2           | 2      | 0       | 6x6             |
| Layer 5 (Conv2d)       | 3           | 1      | 0       | 10x10           |
| Layer 6 (Conv2d)       | 3           | 1      | 0       | 14x14           |
| Layer 7 (AdaptiveAvgPool2d) | -     | -      | -       | 14x14           |
| Layer 8 (Conv2d)       | 1           | 1      | 0       | 14x14           |


### Model 8

#### Target

- Utilize LeakyReLU activation function for improved model performance.
- Adjust the model architecture by adding an additional convolutional block.
- Increase the batch size to 32 for faster training.
- Modify the learning rate scheduler for better convergence.

#### Results

- **Parameters:** 7,792
- **Best Train Accuracy:** 99.41%
- **Best Test Accuracy:** 99.32%

#### Analysis

- The use of LeakyReLU activation function helps in improving the model's performance by mitigating the vanishing gradient problem.
- The addition of an extra convolutional block enhances the model's capacity to learn more complex features.
- Increasing the batch size to 32 allows for faster training while still maintaining good performance.
- The modified learning rate scheduler enables better convergence during the training process.
- The model achieves high accuracy on both the training and test sets, indicating its effectiveness in classification tasks.

#### Receptive Field Calculation

| Layer                  | Kernel Size | Stride | Padding | Receptive Field |
|------------------------|-------------|--------|---------|-----------------|
| Input                  | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)       | 3           | 1      | 0       | 3x3             |
| Layer 2 (Conv2d)       | 3           | 1      | 0       | 5x5             |
| Layer 3 (Conv2d)       | 1           | 1      | 0       | 5x5             |
| Layer 4 (MaxPool2d)    | 2           | 2      | 0       | 6x6             |
| Layer 5 (Conv2d)       | 3           | 1      | 0       | 10x10           |
| Layer 6 (Conv2d)       | 3           | 1      | 0       | 14x14           |
| Layer 7 (Conv2d)       | 3           | 1      | 0       | 18x18           |
| Layer 8 (AdaptiveAvgPool2d) | -     | -      | -       | 18x18           |
| Layer 9 (Conv2d)       | 1           | 1      | 0       | 18x18           |

<img width="680" alt="Pasted Graphic" src="https://github.com/mittalutkarsh/Utkarsh-ERA-V2-Assignment-7/assets/36775837/1340820e-c8ab-4e8d-9b50-842437228ab0">


## Conclusion

Best Model - Model 8
No of Parameters - 7792
Test Accuracy - 99.32
No of Epochs used - 15

## Usage

[Provide instructions on how to run the code and reproduce the results]

