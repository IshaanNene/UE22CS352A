import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Dataset Preparation do not modify, ensure data folder is in the same directory as this file
def prepareData(path):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(64),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


    #In case you want to change path, change it in Test.py do not make a new path variable
    dataset = ImageFolder(path, transform=transform)
    test_size = 0.2
    num_dataset = len(dataset)
    num_test = int(num_dataset * test_size)
    num_train = num_dataset - num_test
    train_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=2)
     

    return train_loader, test_loader

# CNN Model Class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ----------------------------------------------
        # TODO: Define the CNN layers (Conv layers, Pooling layers, Fully connected layers)
        #-----------------------------------------------
        self.setCriterionAndOptimizer()
        


    def forward(self, x):
        # ----------------------------------------------
        # TODO: Define the forward pass through the network.
        # Pass x through convolution layers, pooling layers, flatten, and fully connected layers.
        # ----------------------------------------------
        return x

    def setCriterionAndOptimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)
        self.criterion = nn.CrossEntropyLoss()

# Training Loop
def train(model, train_loader):
    # Implement training loop here
    # Input: train_loader
    # Output: Train accuracy (float)
    # You can set the epochs
    model.train()
    epochs=5
    running_loss=0
    for epoch in range(epochs):
        correct=total=0
        # ----------------------------------------------
        # TODO: Forward pass, calculate loss, backpropagate and step the optimizer
        # ----------------------------------------------
        
        #--------------------------------
        # TODO: Update the running_loss, correct_predictions and total_samples
        #--------------------------------

        
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} - Train Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    print(f'Train Accuracy: {train_accuracy:.2f}%')
    return train_accuracy

# Evaluation Loop
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy

