import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
from data.data_acorn import AcornData
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    """Autoencoder neural network for dimensionality reduction and reconstruction."""
    
    def __init__(self, input_dim: int = 48, encoding_dim: int = 2, hidden_dim: int = 8):
        """
        Initialize the autoencoder model.
        
        Args:
            input_dim: Dimension of input data
            encoding_dim: Dimension of the encoded representation
            hidden_dim: Dimension of the hidden layer
        """
        super(Autoencoder, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input data to the latent space."""
        return self.encoder(x)


def preprocess_data(data: np.ndarray) -> np.ndarray:
    """
    Convert nested array structure to a clean 2D array with consistent dimensions.
    
    Args:
        data: Input data array, potentially with inconsistent dimensions
        
    Returns:
        Processed 2D array with consistent dimensions
    """
    if data.dtype == "object":
        n_samples = data.shape[0]

        lengths = [data[i][0].shape[0] for i in range(n_samples)]
        target_length = max(set(lengths), key=lengths.count)


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


class AutoencoderTrainer:
    """Class to manage the training and evaluation of autoencoder models."""
    
    def __init__(
        self, 
        input_dim: int = 48, 
        encoding_dim: int = 2, 
        hidden_dim: int = 8,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
        model_dir: str = "models/autoencoders"
    ):
        """
        Initialize the trainer with model configuration.
        
        Args:
            input_dim: Dimension of input data
            encoding_dim: Dimension of the encoded representation
            hidden_dim: Dimension of the hidden layer
            learning_rate: Learning rate for optimizer
            device: Device to use for training (auto-detected if None)
            model_dir: Directory to save/load models
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        
        # Set up device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = Autoencoder(
            input_dim=input_dim, 
            encoding_dim=encoding_dim, 
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(
        self, 
        data: np.ndarray, 
        epochs: int = 100, 
        batch_size: int = 256,
        num_workers: int = 4,
        log_interval: int = 10
    ) -> Tuple[nn.Module, List[float]]:
        """
        Train the autoencoder model.
        
        Args:
            data: Input data for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            log_interval: Interval for logging training progress
            
        Returns:
            Trained model and list of losses
        """
        processed_data = preprocess_data(data)
        data_tensor = torch.tensor(processed_data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Training loop
        losses = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in dataloader:
                # Move to device here after DataLoader has processed the batch
                inputs = batch[0].to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                epoch_loss += loss.item()
                batch_count += 1
                
                loss.backward()
                self.optimizer.step()

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count
            losses.append(avg_epoch_loss)
            
            if (epoch + 1) % log_interval == 0:
                logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

        return self.model, losses
    
    def evaluate(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate the model on input data.
        
        Args:
            data: Input data for evaluation
            
        Returns:
            Tuple of reconstructed data and mean squared error
        """
        processed_data = preprocess_data(data)
        data_tensor = torch.tensor(processed_data, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed_data = self.model(data_tensor.to(self.device)).cpu().numpy()
        
        # Calculate reconstruction error
        mse = np.mean(np.square(processed_data - reconstructed_data))
        logging.info(f"Reconstruction Error (MSE): {mse:.6f}")
        
        return reconstructed_data, mse
    
    def visualize_reconstruction(
        self, 
        original_data: np.ndarray, 
        reconstructed_data: np.ndarray, 
        num_examples: int = 5
    ) -> None:
        """
        Visualize original vs reconstructed data examples.
        
        Args:
            original_data: Original input data
            reconstructed_data: Reconstructed data from the model
            num_examples: Number of examples to visualize
        """
        num_examples = min(num_examples, len(original_data))
        plt.figure(figsize=(15, 10))

        for i in range(num_examples):
            plt.subplot(num_examples, 1, i + 1)
            plt.plot(original_data[i], "b-", label="Original")
            plt.plot(reconstructed_data[i], "r-", label="Reconstructed")
            plt.title(f"Example {i + 1}")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self, metadata: Dict[str, Any]) -> str:
        """
        Save the model with metadata information in the filename.
        
        Args:
            metadata: Dictionary containing metadata (acorn_group, years, etc.)
            
        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create filename from metadata
        acorn_group = metadata.get('acorn_group', 'unknown')
        years = metadata.get('selected_years', (0, 0))
        start_year, end_year = years
        
        filename = f"autoencoder_{acorn_group}_{start_year}_{end_year}_{self.encoding_dim}d.pt"
        filepath = os.path.join(self.model_dir, filename)
        
        # Save model state and configuration
        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dim': self.hidden_dim,
            'metadata': metadata
        }
        
        torch.save(state, filepath)
        logging.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, metadata: Dict[str, Any]) -> bool:
        """
        Load a model based on metadata if it exists.
        
        Args:
            metadata: Dictionary containing metadata (acorn_group, years, etc.)
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # Create filename from metadata
        acorn_group = metadata.get('acorn_group', 'unknown')
        years = metadata.get('selected_years', (0, 0))
        start_year, end_year = years
        
        filename = f"autoencoder_{acorn_group}_{start_year}_{end_year}_{self.encoding_dim}d.pt"
        filepath = os.path.join(self.model_dir, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logging.info(f"No existing model found at {filepath}")
            return False
        
        # Load model
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Verify model configuration matches
            if (state['input_dim'] == self.input_dim and 
                state['encoding_dim'] == self.encoding_dim and
                state['hidden_dim'] == self.hidden_dim):
                
                self.model.load_state_dict(state['model_state_dict'])
                logging.info(f"Model loaded from {filepath}")
                return True
            else:
                logging.warning(
                    f"Model configuration mismatch. Found: input_dim={state['input_dim']}, "
                    f"encoding_dim={state['encoding_dim']}, hidden_dim={state['hidden_dim']}"
                )
                return False
                
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False


def train_autoencoder(
    data, input_dim=48, encoding_dim=2, hidden_dim=8, epochs=100, batch_size=256
):
    """Legacy function for backward compatibility."""
    trainer = AutoencoderTrainer(
        input_dim=input_dim, 
        encoding_dim=encoding_dim, 
        hidden_dim=hidden_dim
    )
    model, _ = trainer.train(data, epochs=epochs, batch_size=batch_size)
    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    acorn_group = "Comfortable"
    selected_years = (2011, 2012)
    encoding_dim = 2
    hidden_dim = 8
    
    # Load data
    acorn_data = AcornData(
        acorn_group=acorn_group, 
        selected_years=selected_years
    ).get_data()
    original_data = acorn_data.select("hh_consumption").to_numpy()
    processed_data = preprocess_data(original_data)
    
    # Create metadata dictionary
    metadata = {
        'acorn_group': acorn_group,
        'selected_years': selected_years,
        'data_source': 'acorn',
        'feature': 'hh_consumption'
    }
    
    # Initialize trainer
    trainer = AutoencoderTrainer(
        input_dim=48, 
        encoding_dim=encoding_dim, 
        hidden_dim=hidden_dim,
        learning_rate=0.001
    )
    
    # Try to load existing model
    model_loaded = trainer.load_model(metadata)
    
    # Train if no model was loaded
    if not model_loaded:
        logging.info("Training new model...")
        model, losses = trainer.train(
            original_data,
            epochs=10,
            batch_size=256,
            log_interval=1
        )
        
        trainer.save_model(metadata)
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    else:
        logging.info("Using pre-trained model")
    
    # Evaluate and visualize with loaded or newly trained model
    reconstructed_data, mse = trainer.evaluate(original_data)
    trainer.visualize_reconstruction(processed_data, reconstructed_data, num_examples=5)