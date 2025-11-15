"""
Simple MNIST Digit Recognition with PyTorch
A complete example of training a CNN to recognize handwritten digits 0-9
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for MNIST digit recognition"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data():
    """Load and preprocess MNIST dataset"""
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean/std
        ]
    )

    # Load datasets
    train_dataset = datasets.MNIST(
        root="./mnist_data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./mnist_data", train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """Train the neural network"""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

        print(f"Epoch {epoch + 1} completed")


def evaluate_model(model, test_loader):
    """Evaluate model performance on test set"""
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
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy


def predict_digit(model, image):
    """Predict a single digit from an image"""
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
        return predicted.item()


def visualize_predictions(model, test_dataset, num_images=10):
    """Visualize model predictions on test images"""
    model.eval()

    # Get random test images
    indices = np.random.choice(len(test_dataset), num_images, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = test_dataset[idx]
            predicted = predict_digit(model, image)

            # Denormalize for display
            image_display = image.squeeze().numpy()
            image_display = image_display * 0.3081 + 0.1307  # Denormalize

            axes[i].imshow(image_display, cmap="gray")
            axes[i].set_title(f"Pred: {predicted}, True: {label}")
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the MNIST digit recognition example"""
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data()

    print("Creating model...")
    model = SimpleCNN()

    print("Setting up loss and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, epochs=5)

    print("Evaluating model...")
    accuracy = evaluate_model(model, test_loader)

    print("Visualizing predictions...")
    visualize_predictions(model, test_loader.dataset)

    print("Saving model...")
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as 'mnist_model.pth'")

    return model, accuracy


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(69)
    np.random.seed(69)

    # Run the example
    model, accuracy = main()

    print("Example completed!")
    print(".2f")
    print("You can now use this model to recognize handwritten digits!")
