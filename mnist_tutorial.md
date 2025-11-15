# Simple MNIST Digit Recognition with PyTorch

## Overview
This tutorial demonstrates a simple PyTorch application for recognizing handwritten digits (0-9) using the MNIST dataset. MNIST (Modified National Institute of Standards and Technology) is a classic machine learning dataset containing 70,000 grayscale images of handwritten digits.

## What You'll Learn
- Loading and preprocessing the MNIST dataset
- Building a simple Convolutional Neural Network (CNN)
- Training the model with PyTorch
- Evaluating model performance
- Making predictions on new images

## Prerequisites
```bash
pip install torch torchvision matplotlib numpy
```

## Step 1: Import Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
```

## Step 2: Load and Preprocess Data
```python
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean/std
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## Step 3: Build the Neural Network
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
```

## Step 4: Define Loss and Optimizer
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Step 5: Training Loop
```python
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train(model, train_loader, criterion, optimizer)
```

## Step 6: Evaluate the Model
```python
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

# Evaluate
accuracy = evaluate(model, test_loader)
```

## Step 7: Make Predictions
```python
def predict_digit(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Test on a single image
test_image, test_label = test_dataset[0]
predicted_digit = predict_digit(model, test_image)
print(f'Predicted: {predicted_digit}, Actual: {test_label}')

# Visualize
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted_digit}, Actual: {test_label}')
plt.show()
```

## Step 8: Save and Load Model
```python
# Save model
torch.save(model.state_dict(), 'mnist_model.pth')

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()
```

## Complete Code
See `mnist_recognizer.py` for the complete working example.

## Understanding the Architecture
- **Conv2d**: Convolutional layers extract features from images
- **MaxPool2d**: Reduces spatial dimensions
- **Linear**: Fully connected layers for classification
- **ReLU**: Activation function
- **Dropout**: Prevents overfitting

## Expected Results
With this simple CNN, you should achieve around 98-99% accuracy on the MNIST test set after training for 5 epochs.

## Extensions
- Try different architectures (deeper networks, residual connections)
- Experiment with data augmentation
- Implement early stopping
- Add model checkpoints

## Why PyTorch?
- Dynamic computation graphs
- Easy debugging
- Strong community support
- Seamless GPU acceleration
- Integration with other Python libraries</content>
<parameter name="filePath">D:\projects\PyEEPro\mnist_tutorial.md
