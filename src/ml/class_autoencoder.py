import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.amp as amp
import pickle
from typing import Tuple, Optional, List, Dict, Any
from data.data_acorn import AcornData
from settings import settings
from system_resource_manager import SystemResourceManager


# Configure backend for performance
torch.backends.cudnn.benchmark = True


# =============== Data Processing ===============


class DataProcessor:
    """Class for handling data preprocessing and preparation."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize the data processor.

        Args:
            cache_dir: Directory to use for caching processed data
        """
        self.cache_dir = cache_dir or (settings.CACHE_DIR / "autoencoder_preprocessing")
        os.makedirs(self.cache_dir, exist_ok=True)

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Convert nested array structure to a clean 2D array with consistent dimensions.

        Args:
            data: Input data array, potentially with inconsistent dimensions

        Returns:
            Processed 2D array with consistent dimensions
        """
        # Try to create a cache key from the data
        try:
            if hasattr(data, "shape") and hasattr(data, "dtype"):
                # Simple caching based on shape and basic properties
                cache_key = f"preprocessed_data_{data.shape}_{data.dtype}.pkl"
                cache_path = os.path.join(self.cache_dir, cache_key)

                # Check if cached result exists
                if os.path.exists(cache_path):
                    logging.info(f"Loading preprocessed data from cache: {cache_path}")
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            logging.warning(f"Cache key generation failed: {e}")

        # Process the data if not cached
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
            result = processed_data.astype(np.float32)
        else:
            result = data.astype(np.float32)

        # Save to cache
        try:
            if "cache_path" in locals():
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                logging.info(f"Saved preprocessed data to cache: {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to save to cache: {e}")

        return result

    def create_data_loaders(
        self,
        processed_data: np.ndarray,
        batch_size: int,
        num_workers: int,
        validation_split: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders from processed data.

        Args:
            processed_data: Preprocessed numpy array data
            batch_size: Batch size for training and validation
            num_workers: Number of worker processes for data loading
            validation_split: Fraction of data to use for validation

        Returns:
            Tuple of (train_loader, val_loader)
        """
        indices = np.random.permutation(processed_data.shape[0])
        val_size = int(processed_data.shape[0] * validation_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_data = processed_data[train_indices]
        val_data = processed_data[val_indices]

        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        return train_loader, val_loader

    def create_test_loader(
        self, data: np.ndarray, batch_size: int, num_workers: int
    ) -> DataLoader:
        """
        Create a DataLoader for test/evaluation data.

        Args:
            data: Input data to evaluate
            batch_size: Batch size for evaluation
            num_workers: Number of worker processes for data loading

        Returns:
            DataLoader for test data
        """
        processed_data = self.preprocess_data(data)
        data_tensor = torch.tensor(processed_data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)

        # Create DataLoader for batch processing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return dataloader, processed_data


# =============== Model Classes ===============


class AutoencoderModel(nn.Module):
    """Autoencoder neural network for dimensionality reduction and reconstruction."""

    def __init__(self, input_dim: int = 48, encoding_dim: int = 2, hidden_dim: int = 8):
        """
        Initialize the autoencoder model.

        Args:
            input_dim: Dimension of input data
            encoding_dim: Dimension of the encoded representation
            hidden_dim: Dimension of the hidden layer
        """
        super(AutoencoderModel, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim

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


class AutoencoderTrainer:
    """Class to manage the training and evaluation of autoencoder models."""

    def __init__(
        self,
        input_dim: int = 48,
        encoding_dim: int = 2,
        hidden_dim: int = 8,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
        model_dir: str = settings.MODEL_DIR,
        use_amp: bool = True,
        auto_resource_adjustment: bool = True,
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
            use_amp: Whether to use Automatic Mixed Precision for faster training
            auto_resource_adjustment: Whether to automatically adjust settings based on system resources
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.auto_resource_adjustment = auto_resource_adjustment

        # Initialize components
        self.resource_manager = SystemResourceManager()
        self.data_processor = DataProcessor()

        # Get system resources
        self.system_resources = self.resource_manager.get_system_resources()

        # Set up device
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logging.info(f"Using device: {self.device}")

        # Log system resources
        logging.info(
            f"System resources: CPU cores: {self.system_resources['cpu_count']}, "
            f"Physical cores: {self.system_resources['cpu_count_physical']}"
        )

        if torch.cuda.is_available():
            gpu_mem_gb = self.system_resources["cuda_memory_total"] / (1024**3)
            logging.info(
                f"GPU: {self.system_resources['cuda_device_name']}, "
                f"Memory: {gpu_mem_gb:.2f} GB"
            )

            # Adjust model complexity based on GPU memory if auto-adjustment is enabled
            if self.auto_resource_adjustment:
                # For very low memory GPUs, reduce hidden dimension
                if gpu_mem_gb < 2:
                    self.hidden_dim = min(self.hidden_dim, 4)
                    logging.info(
                        f"Limited GPU memory detected. Reduced hidden_dim to {self.hidden_dim}"
                    )

        # Initialize model
        self.model = AutoencoderModel(
            input_dim=input_dim, encoding_dim=encoding_dim, hidden_dim=hidden_dim
        ).to(self.device)

        # Calculate model parameters
        self.model_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Model parameters: {self.model_params:,}")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Determine if we should use AMP
        self.use_amp = use_amp and torch.cuda.is_available()

        # Initialize scaler for mixed precision training
        self.scaler = (
            amp.GradScaler("cuda", enabled=self.use_amp)
            if torch.cuda.is_available()
            else amp.GradScaler(enabled=False)
        )

    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        log_interval: int = 10,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        gradient_accumulation_steps: int = 1,
    ) -> Tuple[nn.Module, List[float]]:
        """
        Train the autoencoder model.

        Args:
            data: Input data for training
            epochs: Number of training epochs
            batch_size: Batch size for training (auto-determined if None)
            num_workers: Number of worker processes for data loading (auto-determined if None)
            log_interval: Interval for logging training progress
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights

        Returns:
            Trained model and list of losses
        """
        start_time = time.time()
        processed_data = self.data_processor.preprocess_data(data)
        logging.info(f"Data preprocessing took {time.time() - start_time:.2f} seconds")

        # Set up adaptive resource parameters
        if self.auto_resource_adjustment:
            if num_workers is None:
                num_workers = self.resource_manager.calculate_optimal_workers(
                    self.system_resources["cpu_count"]
                )
                logging.info(f"Auto-configured worker processes: {num_workers}")

            # Determine optimal batch size
            if batch_size is None:
                if torch.cuda.is_available():
                    available_memory = (
                        self.system_resources["cuda_memory_total"]
                        - self.system_resources["cuda_memory_reserved"]
                    )
                    precision = "mixed" if self.use_amp else "full"
                    batch_size = self.resource_manager.calculate_optimal_batch_size(
                        self.input_dim, self.model_params, available_memory, precision
                    )
                else:
                    # For CPU, use smaller batch sizes based on available RAM
                    memory_gb = self.system_resources["memory_available"] / (1024**3)
                    batch_size = min(64, max(8, int(memory_gb * 4)))

                logging.info(f"Auto-configured batch size: {batch_size}")

                # Adjust gradient accumulation for effective larger batch sizes
                if batch_size < 128 and processed_data.shape[0] > 10000:
                    gradient_accumulation_steps = max(1, 128 // batch_size)
                    logging.info(
                        f"Using gradient accumulation: {gradient_accumulation_steps} steps"
                    )
        else:
            if batch_size is None:
                batch_size = 256
            if num_workers is None:
                num_workers = 4

        # Create data loaders
        train_loader, val_loader = self.data_processor.create_data_loaders(
            processed_data, batch_size, num_workers, validation_split
        )

        # Training loop
        losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            # Reset optimizer gradients at the beginning of each epoch
            self.optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                # Move to device
                inputs = batch[0].to(self.device)

                # Use mixed precision if enabled
                if self.use_amp:
                    with amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                        # Scale loss by accumulation steps for consistent gradients
                        loss = loss / gradient_accumulation_steps

                    # Scale gradients and accumulate
                    self.scaler.scale(loss).backward()

                    # Only update weights after accumulating gradients
                    if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(
                        train_loader
                    ):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # Regular full-precision training
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)
                    # Scale loss by accumulation steps
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    # Only update weights after accumulating gradients
                    if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(
                        train_loader
                    ):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # Track loss (multiply back by accumulation steps for reporting)
                epoch_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count
            losses.append(avg_epoch_loss)

            # Validation phase
            self.model.eval()
            val_epoch_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(self.device)

                    if self.use_amp:
                        with amp.autocast(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        ):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, inputs)
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)

                    val_epoch_loss += loss.item()
                    val_batch_count += 1

            avg_val_loss = val_epoch_loss / val_batch_count
            val_losses.append(avg_val_loss)

            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(avg_val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]

            if old_lr != new_lr:
                logging.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            if (epoch + 1) % log_interval == 0:
                logging.info(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Train Loss: {avg_epoch_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

        self.plot_training_history(losses, val_losses)

        return self.model, losses

    def plot_training_history(
        self, losses: List[float], val_losses: List[float]
    ) -> None:
        """
        Plot training and validation loss history.

        Args:
            losses: List of training losses
            val_losses: List of validation losses
        """
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(
        self, data: np.ndarray, batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate the model on input data.

        Args:
            data: Input data for evaluation
            batch_size: Batch size for evaluation (auto-determined if None)

        Returns:
            Tuple of reconstructed data and mean squared error
        """
        # Auto-determine batch size if needed
        if batch_size is None and self.auto_resource_adjustment:
            if torch.cuda.is_available():
                available_memory = (
                    self.system_resources["cuda_memory_total"]
                    - self.system_resources["cuda_memory_reserved"]
                )
                precision = "mixed" if self.use_amp else "full"
                batch_size = self.resource_manager.calculate_optimal_batch_size(
                    self.input_dim, self.model_params, available_memory, precision
                )
                # Use larger batches for inference
                batch_size = batch_size * 2
            else:
                # For CPU, estimate based on available memory
                memory_gb = self.system_resources["memory_available"] / (1024**3)
                batch_size = min(128, max(16, int(memory_gb * 8)))
        elif batch_size is None:
            batch_size = 512

        logging.info(f"Evaluation batch size: {batch_size}")

        # Determine optimal workers for evaluation
        if self.auto_resource_adjustment:
            num_workers = self.resource_manager.calculate_optimal_workers(
                self.system_resources["cpu_count"]
            )
        else:
            num_workers = 4

        # Create DataLoader for test data
        dataloader, processed_data = self.data_processor.create_test_loader(
            data, batch_size, num_workers
        )

        self.model.eval()
        reconstructed_chunks = []

        with torch.no_grad():
            if self.use_amp:
                with amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    for batch in dataloader:
                        inputs = batch[0].to(self.device)
                        outputs = self.model(inputs)
                        reconstructed_chunks.append(outputs.cpu().numpy())
            else:
                for batch in dataloader:
                    inputs = batch[0].to(self.device)
                    outputs = self.model(inputs)
                    reconstructed_chunks.append(outputs.cpu().numpy())

        # Concatenate all batches
        reconstructed_data = np.vstack(reconstructed_chunks)

        # Calculate reconstruction error
        mse = np.mean(np.square(processed_data - reconstructed_data))
        logging.info(f"Reconstruction Error (MSE): {mse:.6f}")

        return reconstructed_data, mse

    def visualize_reconstruction(
        self,
        original_data: np.ndarray,
        reconstructed_data: np.ndarray,
        num_examples: int = 5,
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
        acorn_group = metadata.get("acorn_group", "unknown")
        years = metadata.get("selected_years", (0, 0))
        start_year, end_year = years

        filename = (
            f"autoencoder_{acorn_group}_{start_year}_{end_year}_{self.encoding_dim}d.pt"
        )
        filepath = os.path.join(self.model_dir, filename)

        # Save model state and configuration
        state = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dim": self.hidden_dim,
            "metadata": metadata,
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
        acorn_group = metadata.get("acorn_group", "unknown")
        years = metadata.get("selected_years", (0, 0))
        start_year, end_year = years

        filename = (
            f"autoencoder_{acorn_group}_{start_year}_{end_year}_{self.encoding_dim}d.pt"
        )
        filepath = os.path.join(self.model_dir, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            logging.info(f"No existing model found at {filepath}")
            return False

        # Load model
        try:
            state = torch.load(filepath, map_location=self.device)

            # Verify model configuration matches
            if (
                state["input_dim"] == self.input_dim
                and state["encoding_dim"] == self.encoding_dim
                and state["hidden_dim"] == self.hidden_dim
            ):
                self.model.load_state_dict(state["model_state_dict"])
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


# =============== Pipeline Class ===============


class AutoencoderPipeline:
    """End-to-end pipeline for autoencoder workflow."""

    def __init__(
        self,
        acorn_group: str = "Comfortable",
        selected_years: Tuple[int, int] = (2011, 2012),
        encoding_dim: int = 2,
        hidden_dim: int = 8,
        learning_rate: float = 0.001,
        auto_resource_adjustment: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the autoencoder pipeline.

        Args:
            acorn_group: Acorn group to use for data
            selected_years: Year range to select data from
            encoding_dim: Dimension of the encoded representation
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for optimizer
            auto_resource_adjustment: Whether to automatically adjust resources
            device: Device to use (auto-detected if None)
        """
        self.acorn_group = acorn_group
        self.selected_years = selected_years
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.auto_resource_adjustment = auto_resource_adjustment

        # Set device if not provided
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Metadata for model identification
        self.metadata = {
            "acorn_group": acorn_group,
            "selected_years": selected_years,
            "data_source": "acorn",
            "feature": "hh_consumption",
        }

        # Will be set when data is loaded
        self.input_dim = None
        self.original_data = None
        self.processed_data = None

        # Initialize components
        self.data_processor = DataProcessor()

    def load_data(self):
        """Load and preprocess data from the Acorn dataset."""
        start_time = time.time()
        acorn_data = AcornData(
            acorn_group=self.acorn_group, selected_years=self.selected_years
        ).get_data()
        self.original_data = acorn_data.select("hh_consumption").to_numpy()
        logging.info(f"Data loading took {time.time() - start_time:.2f} seconds")

        # Process data
        self.processed_data = self.data_processor.preprocess_data(self.original_data)
        self.input_dim = (
            self.processed_data.shape[1]
            if self.processed_data.ndim > 1
            else self.processed_data.shape[0]
        )

        # Initialize the trainer with the correct input dimension
        self.trainer = AutoencoderTrainer(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            use_amp=True,
            auto_resource_adjustment=self.auto_resource_adjustment,
            device=self.device,
        )

        return self.processed_data

    def train_or_load_model(self, force_retrain: bool = False, epochs: int = 20):
        """Train a new model or load a pre-trained one."""
        if not hasattr(self, "trainer"):
            raise ValueError(
                "Data must be loaded before training or loading a model. Call load_data() first."
            )

        if not force_retrain:
            model_loaded = self.trainer.load_model(self.metadata)
        else:
            model_loaded = False

        if not model_loaded:
            logging.info("Training new model...")
            self.trainer.train(
                self.original_data,
                epochs=epochs,
                batch_size=None,
                num_workers=None,
                log_interval=1,
                validation_split=0.1,
                early_stopping_patience=5,
            )

            self.trainer.save_model(self.metadata)
            return False
        else:
            logging.info("Using pre-trained model")
            return True

    def evaluate_model(self):
        """Evaluate the model on the loaded data."""
        if not hasattr(self, "trainer"):
            raise ValueError(
                "Model must be trained or loaded before evaluation. Call train_or_load_model() first."
            )

        start_time = time.time()
        reconstructed_data, mse = self.trainer.evaluate(
            self.original_data, batch_size=None
        )
        logging.info(f"Evaluation took {time.time() - start_time:.2f} seconds")

        return reconstructed_data, mse

    def visualize_results(self, reconstructed_data, num_examples=5):
        """Visualize the reconstruction results."""
        if not hasattr(self, "trainer"):
            raise ValueError(
                "Model must be evaluated before visualization. Call evaluate_model() first."
            )

        self.trainer.visualize_reconstruction(
            self.processed_data, reconstructed_data, num_examples=num_examples
        )

    def run_pipeline(
        self, force_retrain: bool = False, epochs: int = 20, num_examples: int = 5
    ):
        """Run the full pipeline: load data, train/load model, evaluate, and visualize."""
        # Load and preprocess data
        self.load_data()

        # Train or load model
        self.train_or_load_model(force_retrain=force_retrain, epochs=epochs)

        # Evaluate model
        reconstructed_data, mse = self.evaluate_model()

        # Visualize results
        self.visualize_results(reconstructed_data, num_examples=num_examples)

        return reconstructed_data, mse


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the full pipeline
    pipeline = AutoencoderPipeline(
        acorn_group="Comfortable",
        selected_years=(2011, 2012),
        encoding_dim=2,
        hidden_dim=8,
        auto_resource_adjustment=True
    )

    pipeline.run_pipeline(force_retrain=False, epochs=20, num_examples=5)
