import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import numpy as np
import pickle
from typing import Tuple
from settings import settings


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