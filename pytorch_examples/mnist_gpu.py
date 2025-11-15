import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./mnist_data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./mnist_data", train=False, download=True, transform=transform
    )

    # GPU: Larger batch size and pin_memory for faster GPU transfers
    batch_size = 128 if torch.cuda.is_available() else 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # GPU: Move data to device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

        print(f"Epoch {epoch + 1} completed")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # GPU: Move data to device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy


def predict_digit(model, image, device):
    model.eval()
    with torch.no_grad():
        # GPU: Move image to device
        image = image.to(device)
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        return predicted.item()


def visualize_predictions(model, test_dataset, device, num_images=10):
    model.eval()

    indices = np.random.choice(len(test_dataset), num_images, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = test_dataset[idx]
            predicted = predict_digit(model, image, device)

            image_display = image.squeeze().numpy()
            image_display = image_display * 0.3081 + 0.1307

            axes[i].imshow(image_display, cmap="gray")
            axes[i].set_title(f"Pred: {predicted}, True: {label}")
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # GPU: Set up device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data()

    print("Creating model...")
    model = SimpleCNN()
    # GPU: Move model to device
    model.to(device)

    print("Setting up loss and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    print("Evaluating model...")
    accuracy = evaluate_model(model, test_loader, device)

    print("Visualizing predictions...")
    visualize_predictions(model, test_loader.dataset, device)

    print("Saving model...")
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as 'mnist_model.pth'")

    return model, accuracy


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # GPU: Enable cuDNN benchmark for optimized performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model, accuracy = main()

    print("Example completed!")
    print(f"Final accuracy: {accuracy:.2f}%")
    print("You can now use this model to recognize handwritten digits!")
