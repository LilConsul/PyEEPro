import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from data.data_acorn import AcornData
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, input_dim=48, encoding_dim=2, hidden_dim=8):
        super(Autoencoder, self).__init__()
        # More modular structure with configurable dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def preprocess_data(data):
    """Convert nested array structure to a clean 2D array with consistent dimensions."""
    if data.dtype == "object":
        n_samples = data.shape[0]

        lengths = [data[i][0].shape[0] for i in range(n_samples)]
        target_length = max(set(lengths), key=lengths.count)

        print(f"Target length: {target_length}, unique lengths found: {set(lengths)}")

        processed_arrays = []
        for i in range(n_samples):
            arr = data[i][0]
            curr_len = arr.shape[0]

            if curr_len == target_length:
                processed_arrays.append(arr)
            elif curr_len > target_length:
                # Truncate longer arrays
                processed_arrays.append(arr[:target_length])
            else:
                # Pad shorter arrays with zeros
                padded = np.zeros(target_length, dtype=np.float32)
                padded[:curr_len] = arr
                processed_arrays.append(padded)

        processed_data = np.vstack(processed_arrays)
        return processed_data.astype(np.float32)

    return data.astype(np.float32)


def train_autoencoder(
    data, input_dim=48, encoding_dim=2, hidden_dim=8, epochs=100, batch_size=256
):
    processed_data = preprocess_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_tensor = torch.tensor(processed_data, dtype=torch.float32)

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallelize data loading
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Create model and move to device
    model = Autoencoder(
        input_dim=input_dim, encoding_dim=encoding_dim, hidden_dim=hidden_dim
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    return model

if __name__ == "__main__":
    acorn_data = AcornData(
        acorn_group="Comfortable", selected_years=(2011, 2012)
    ).get_data()

    original_data = acorn_data.select("hh_consumption").to_numpy()

    # Train the model
    model = train_autoencoder(
        original_data,
        input_dim=48,
        encoding_dim=2,
        hidden_dim=8,
        epochs=10,
        batch_size=32,
    )

    # Get reconstructed data
    model.eval()
    with torch.no_grad():
        processed_data = preprocess_data(original_data)
        data_tensor = torch.tensor(processed_data, dtype=torch.float32)
        device = next(model.parameters()).device  # Get model's device
        reconstructed_data = model(data_tensor.to(device)).cpu().numpy()

    # Plot original vs reconstructed data
    num_examples = min(5, len(processed_data))
    plt.figure(figsize=(15, 10))

    for i in range(num_examples):
        plt.subplot(num_examples, 1, i + 1)
        plt.plot(processed_data[i], "b-", label="Original")
        plt.plot(reconstructed_data[i], "r-", label="Reconstructed")
        plt.title(f"Example {i + 1}")
        plt.legend()
        plt.grid(True)

    # Calculate reconstruction error
    mse = np.mean(np.square(processed_data - reconstructed_data))
    print(f"Reconstruction Error (MSE): {mse:.6f}")
    plt.tight_layout()
    plt.show()