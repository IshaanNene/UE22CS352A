import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Dataset Preparation
def prepareData(path):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(64),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
        
    # In case you want to change path, change it in Test.py do not make a new path variable
    train_path = path + "train_images"
    test_path = path + "test_images"
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)

    return train_loader, test_loader

# CNN Model Class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 64)  
        self.fc2 = nn.Linear(64, 10)  
        self.setCriterionAndOptimizer()
        


    def forward(self, x):
        #To do: Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

    def setCriterionAndOptimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

# Training Loop
def train(model, train_loader):
    model.train()
    epochs=5
    running_loss=0
    for epoch in range(epochs):
        total, correct = 0, 0

        for inputs, labels in train_loader:
            model.optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()  # Backpropagation
            model.optimizer.step()

            # Accuracy calculation
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            

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




