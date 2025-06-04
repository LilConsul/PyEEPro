import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import logging
import os
import platform
import psutil
from typing import Tuple, Optional, List, Dict, Any
from data.data_acorn import AcornData
import numpy as np
import matplotlib.pyplot as plt
import torch.amp as amp
import pickle
from settings import settings

torch.backends.cudnn.benchmark = True


def get_system_resources() -> Dict[str, Any]:
    """
    Get information about available system resources.
    
    Returns:
        Dictionary with system resource information
    """
    resources = {
        'cpu_count': os.cpu_count(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_percent': psutil.cpu_percent(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'platform': platform.system(),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if resources['cuda_available']:
        resources.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory,
            'cuda_memory_reserved': torch.cuda.memory_reserved(0),
            'cuda_memory_allocated': torch.cuda.memory_allocated(0),
        })
        
    return resources


def calculate_optimal_workers(total_cores: int) -> int:
    """
    Calculate the optimal number of worker processes for data loading.
    
    Args:
        total_cores: Total number of CPU cores available
        
    Returns:
        Optimal number of worker processes
    """
    # Reserve cores for system and main process
    if total_cores <= 2:
        return 0  # Disable multiprocessing for systems with few cores
    elif total_cores <= 4:
        return max(1, total_cores - 1)  # Reserve 1 core
    else:
        # For systems with many cores, use 75% of cores, rounding down
        return max(1, int(total_cores * 0.75))


def calculate_optimal_batch_size(
    input_dim: int, 
    model_params: int, 
    available_memory: int,
    precision: str = 'mixed'
) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        input_dim: Dimension of input data
        model_params: Number of model parameters
        available_memory: Available memory in bytes
        precision: Precision mode ('full' or 'mixed')
        
    Returns:
        Optimal batch size
    """
    # Estimate bytes per sample based on precision
    bytes_per_float = 2 if precision == 'mixed' else 4
    
    # Memory for input, output, gradients, optimizer states, etc.
    bytes_per_sample = input_dim * 4 * bytes_per_float  # Input, output, gradients, backward pass
    
    # Model memory (parameters, gradients, optimizer states)
    model_memory = model_params * 4 * bytes_per_float * 3  # Weights, gradients, optimizer states
    
    # Use 70% of available memory, accounting for model memory and other overhead
    usable_memory = (available_memory * 0.7) - model_memory
    
    # Calculate batch size
    batch_size = max(1, int(usable_memory / bytes_per_sample))
    
    # Cap at reasonable values and ensure it's a power of 2 for GPU efficiency
    batch_size = min(batch_size, 8192)
    
    # Round down to the nearest power of 2 for better GPU utilization
    batch_size = 2 ** int(np.log2(batch_size))
    
    return batch_size


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
    # Create a unique key for caching
    cache_dir = settings.CACHE_DIR / "autoencoder_preprocessing"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Try to create a cache key from the data
    try:
        if hasattr(data, 'shape') and hasattr(data, 'dtype'):
            # Simple caching based on shape and basic properties
            cache_key = f"preprocessed_data_{data.shape}_{data.dtype}.pkl"
            cache_path = os.path.join(cache_dir, cache_key)
            
            # Check if cached result exists
            if os.path.exists(cache_path):
                logging.info(f"Loading preprocessed data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
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
        if 'cache_path' in locals():
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logging.info(f"Saved preprocessed data to cache: {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to save to cache: {e}")
        
    return result


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
        auto_resource_adjustment: bool = True
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
        
        # Get system resources
        self.system_resources = get_system_resources()
        
        # Set up device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Using device: {self.device}")
        
        # Log system resources
        logging.info(f"System resources: CPU cores: {self.system_resources['cpu_count']}, "
                   f"Physical cores: {self.system_resources['cpu_count_physical']}")
        
        if torch.cuda.is_available():
            gpu_mem_gb = self.system_resources['cuda_memory_total'] / (1024**3)
            logging.info(f"GPU: {self.system_resources['cuda_device_name']}, "
                       f"Memory: {gpu_mem_gb:.2f} GB")
            
            # Adjust model complexity based on GPU memory if auto-adjustment is enabled
            if self.auto_resource_adjustment:
                # For very low memory GPUs, reduce hidden dimension
                if gpu_mem_gb < 2:
                    self.hidden_dim = min(self.hidden_dim, 4)
                    logging.info(f"Limited GPU memory detected. Reduced hidden_dim to {self.hidden_dim}")
        
        # Initialize model
        self.model = Autoencoder(
            input_dim=input_dim, 
            encoding_dim=encoding_dim, 
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Calculate model parameters
        self.model_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Model parameters: {self.model_params:,}")
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Determine if we should use AMP
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Initialize scaler for mixed precision training
        self.scaler = amp.GradScaler('cuda', enabled=self.use_amp) if torch.cuda.is_available() else amp.GradScaler(enabled=False)
    
    def train(
        self, 
        data: np.ndarray, 
        epochs: int = 100, 
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        log_interval: int = 10,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        gradient_accumulation_steps: int = 1
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
        processed_data = preprocess_data(data)
        logging.info(f"Data preprocessing took {time.time() - start_time:.2f} seconds")
        
        indices = np.random.permutation(processed_data.shape[0])
        val_size = int(processed_data.shape[0] * validation_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_data = processed_data[train_indices]
        val_data = processed_data[val_indices]
        
        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
        
        if self.auto_resource_adjustment:
            if num_workers is None:
                num_workers = calculate_optimal_workers(self.system_resources['cpu_count'])
                logging.info(f"Auto-configured worker processes: {num_workers}")
            
            # Determine optimal batch size
            if batch_size is None:
                if torch.cuda.is_available():
                    available_memory = (self.system_resources['cuda_memory_total'] - 
                                       self.system_resources['cuda_memory_reserved'])
                    precision = 'mixed' if self.use_amp else 'full'
                    batch_size = calculate_optimal_batch_size(
                        self.input_dim, 
                        self.model_params, 
                        available_memory,
                        precision
                    )
                else:
                    # For CPU, use smaller batch sizes based on available RAM
                    memory_gb = self.system_resources['memory_available'] / (1024**3)
                    batch_size = min(64, max(8, int(memory_gb * 4)))
                
                logging.info(f"Auto-configured batch size: {batch_size}")
                
                # Adjust gradient accumulation for effective larger batch sizes
                if batch_size < 128 and processed_data.shape[0] > 10000:
                    gradient_accumulation_steps = max(1, 128 // batch_size)
                    logging.info(f"Using gradient accumulation: {gradient_accumulation_steps} steps")
        else:
            if batch_size is None:
                batch_size = 256
            if num_workers is None:
                num_workers = 4
        
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
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Training loop
        losses = []
        val_losses = []
        best_val_loss = float('inf')
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
                    with amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                        # Scale loss by accumulation steps for consistent gradients
                        loss = loss / gradient_accumulation_steps
                    
                    # Scale gradients and accumulate
                    self.scaler.scale(loss).backward()
                    
                    # Only update weights after accumulating gradients
                    if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
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
                    if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
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
                        with amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, inputs)
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                    
                    val_epoch_loss += loss.item()
                    val_batch_count += 1
            
            avg_val_loss = val_epoch_loss / val_batch_count
            val_losses.append(avg_val_loss)
            
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(avg_val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                logging.info(f'Learning rate changed from {old_lr:.6f} to {new_lr:.6f}')
            
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

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        return self.model, losses
    
    def evaluate(self, data: np.ndarray, batch_size: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Evaluate the model on input data.
        
        Args:
            data: Input data for evaluation
            batch_size: Batch size for evaluation (auto-determined if None)
            
        Returns:
            Tuple of reconstructed data and mean squared error
        """
        processed_data = preprocess_data(data)
        
        # Auto-determine batch size if needed
        if batch_size is None and self.auto_resource_adjustment:
            if torch.cuda.is_available():
                available_memory = (self.system_resources['cuda_memory_total'] - 
                                   self.system_resources['cuda_memory_reserved'])
                precision = 'mixed' if self.use_amp else 'full'
                batch_size = calculate_optimal_batch_size(
                    self.input_dim, 
                    self.model_params, 
                    available_memory,
                    precision
                )
                # Use larger batches for inference
                batch_size = batch_size * 2
            else:
                # For CPU, estimate based on available memory
                memory_gb = self.system_resources['memory_available'] / (1024**3)
                batch_size = min(128, max(16, int(memory_gb * 8)))
        elif batch_size is None:
            batch_size = 512
            
        logging.info(f"Evaluation batch size: {batch_size}")
        
        data_tensor = torch.tensor(processed_data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)
        
        # Determine optimal workers for evaluation
        if self.auto_resource_adjustment:
            num_workers = calculate_optimal_workers(self.system_resources['cpu_count'])
        else:
            num_workers = 4
        
        # Create DataLoader for batch processing
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
        self.model.eval()
        reconstructed_chunks = []
        
        with torch.no_grad():
            if self.use_amp:
                with amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
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
    
    acorn_group = "Comfortable"
    selected_years = (2011, 2012)
    encoding_dim = 2
    hidden_dim = 8
    
    start_time = time.time()
    acorn_data = AcornData(
        acorn_group=acorn_group, 
        selected_years=selected_years
    ).get_data()
    original_data = acorn_data.select("hh_consumption").to_numpy()
    logging.info(f"Data loading took {time.time() - start_time:.2f} seconds")
    
    processed_data = preprocess_data(original_data)
    
    metadata = {
        'acorn_group': acorn_group,
        'selected_years': selected_years,
        'data_source': 'acorn',
        'feature': 'hh_consumption'
    }
    
    trainer = AutoencoderTrainer(
        input_dim=48, 
        encoding_dim=encoding_dim, 
        hidden_dim=hidden_dim,
        learning_rate=0.001,
        use_amp=True,
        auto_resource_adjustment=True
    )
    
    model_loaded = trainer.load_model(metadata)
    
    if not model_loaded:
        logging.info("Training new model...")
        model, losses = trainer.train(
            original_data,
            epochs=20,
            batch_size=None,
            num_workers=None,
            log_interval=1,
            validation_split=0.1,
            early_stopping_patience=5
        )
        
        trainer.save_model(metadata)
    else:
        logging.info("Using pre-trained model")
    
    start_time = time.time()
    reconstructed_data, mse = trainer.evaluate(original_data, batch_size=None)
    logging.info(f"Evaluation took {time.time() - start_time:.2f} seconds")
    
    trainer.visualize_reconstruction(processed_data, reconstructed_data, num_examples=5)