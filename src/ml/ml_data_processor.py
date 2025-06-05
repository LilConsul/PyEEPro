import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import numpy as np
import pickle
from typing import Tuple
from settings import settings
import polars as pl


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

    def preprocess_conditions(self, conditions):
        """
        Process condition data for conditional autoencoder using Polars.

        Args:
            conditions: DataFrame with condition columns

        Returns:
            Processed condition tensor ready for model input
        """
        # Convert to polars if it's not already
        if not isinstance(conditions, pl.DataFrame):
            processed = pl.from_pandas(conditions)
        else:
            processed = conditions.clone()

        # Handle string boolean values in is_weekend
        if "is_weekend" in processed.columns:
            processed = processed.with_columns(
                pl.when(pl.col("is_weekend") == "true")
                .then(1)
                .when(pl.col("is_weekend") == "false")
                .then(0)
                .otherwise(pl.col("is_weekend"))
                .alias("is_weekend")
            )

        # Adjust day_of_week values to be 0-based if they're 1-based (1-7)
        if "day_of_week" in processed.columns:
            min_day = processed["day_of_week"].min()
            max_day = processed["day_of_week"].max()

            if min_day >= 1 and max_day <= 7:
                processed = processed.with_columns(
                    (pl.col("day_of_week") - 1).alias("day_of_week")
                )

        # One-hot encode categorical features
        categorical_features = {
            "day_of_week": 7,  # 0-6
            "is_weekend": 2,  # 0/1
            "season": 4,  # Assuming 4 seasons
        }

        one_hot_encoded = {}
        for feat, num_classes in categorical_features.items():
            if feat in processed.columns:
                # One-hot encode
                feat_encoded = np.zeros((len(processed), num_classes))
                for i, val in enumerate(processed[feat].to_numpy()):
                    # Safely handle values by ensuring they're within valid range
                    val_idx = min(max(0, int(val)), num_classes - 1)
                    feat_encoded[i, val_idx] = 1
                one_hot_encoded[feat] = feat_encoded

        # Normalize numerical features
        numerical_features = ["avg_temperature", "avg_humidity"]
        normalized_numericals = {}

        for feat in numerical_features:
            if feat in processed.columns:
                values = processed[feat].cast(pl.Float64).to_numpy().reshape(-1, 1)
                # Fill NaN values with mean or 0
                has_nan = np.isnan(values).any()
                if has_nan:
                    values = np.nan_to_num(
                        values,
                        nan=np.nanmean(values)
                        if not np.isnan(np.nanmean(values))
                        else 0,
                    )

                mean = np.mean(values)
                std = np.std(values)
                # Avoid division by zero
                if std < 1e-7:
                    normalized_numericals[feat] = (
                        values - mean
                    )  # Just center if no variance
                else:
                    normalized_numericals[feat] = (values - mean) / std

        # Combine all processed features
        feature_parts = []

        # Add features in consistent order
        for feat in categorical_features:
            if feat in one_hot_encoded:
                feature_parts.append(one_hot_encoded[feat])

        for feat in numerical_features:
            if feat in normalized_numericals:
                feature_parts.append(normalized_numericals[feat])

        processed_conditions = np.concatenate(feature_parts, axis=1)

        return processed_conditions.astype(np.float32)

    def create_data_loaders(
        self,
        processed_data: np.ndarray,
        processed_conditions: np.ndarray,
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
        train_conditions = processed_conditions[train_indices]
        val_data = processed_data[val_indices]
        val_conditions = processed_conditions[val_indices]

        train_dataset = TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_conditions, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(val_data, dtype=torch.float32),
            torch.tensor(val_conditions, dtype=torch.float32),
        )

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
        self,
        data: np.ndarray,
        conditions: np.ndarray,
        batch_size: int,
        num_workers: int,
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
        data_tensor = torch.tensor(data, dtype=torch.float32)
        condition_tensor = torch.tensor(conditions, dtype=torch.float32)

        dataset = TensorDataset(data_tensor, condition_tensor)

        # Create DataLoader for batch processing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return dataloader
