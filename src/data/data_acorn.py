import polars as pl
import numpy as np
import os
import pickle
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from settings import settings
from data.cache_manager import CacheManager


class AcornData:
    """
    Class for loading and processing Acorn data.

    Handles:
    - Loading data from CSV files
    - Data preprocessing
    - Creating data loaders for machine learning

    Features include:
    - day_of_week (0–6),
    - is_weekend (bool),
    - avg_temperature (float),
    - avg_humidity(float),
    - season (int),
    - hh_consumption (list[float])
    """

    def __init__(self, acorn_group, selected_years: (int, int) = (2011, 2021)):
        """
        Initializes the AcornData class with a specific acorn group.

        Args:
            acorn_group: The acorn group to filter data by
            selected_years: Tuple of (start_year, end_year) to filter data
        """
        self.cache_manager = CacheManager(cache_dir=settings.CACHE_DIR / "acorn_data")
        self.acorn_group = acorn_group
        self.files = (
            pl.read_csv(settings.INFORMATION_HOUSEHOLD_FILE)
            .filter(pl.col("Acorn_grouped") == self.acorn_group)
            .select(pl.col("file"))
            .unique()
        )
        if self.files.is_empty():
            raise ValueError(f"No data found for Acorn group: {self.acorn_group}")
        self.selected_years = selected_years
        self._weather_data = pl.read_csv(settings.WEATHER_DAILY_FILE)
        self._holidays_data = pl.read_csv(settings.HOLIDAYS_FILE)

        # Processing cache directory
        self.processor_cache_dir = settings.CACHE_DIR / "autoencoder_preprocessing"
        os.makedirs(self.processor_cache_dir, exist_ok=True)

    def get_data(self) -> pl.DataFrame:
        """
        Retrieves the processed data for the specified acorn group.
        :return: Polars DataFrame with processed data.
        """
        cache_file = f"acorn_data_{self.acorn_group}_({self.selected_years[0]}-{self.selected_years[1]}).csv"
        cached_data = self.cache_manager.load_cache(cache_file)
        if cached_data is not None:
            return cached_data

        df = self._process_data()
        self.cache_manager.save_cache(df, cache_file)
        return df

    def _process_data(self) -> pl.DataFrame:
        result_frames = []

        weather_daily = (
            self._weather_data.with_columns(
                pl.col("time")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                .dt.date()
                .alias("date")
            )
            .group_by("date")
            .agg(
                [
                    pl.mean("temperatureMax").alias("avg_temperature"),
                    pl.mean("humidity").alias("avg_humidity"),
                ]
            )
        )

        holidays_df = (
            self._holidays_data.with_columns(
                pl.col("Bank holidays").str.to_date().alias("holiday_date")
            )
            .select("holiday_date")
            .with_columns(pl.lit(True).alias("is_holiday"))
        )

        for file in self.files["file"].to_list():
            file_path = settings.HHBLOCKS_DIR / (file + ".csv")
            if not file_path.exists():
                raise FileNotFoundError(
                    f"File {file} does not exist in {settings.HHBLOCKS_DIR}"
                )
            if settings.DEBUG:
                print(f"Processing file: {file_path}")

            df = pl.read_csv(file_path)
            df = df.with_columns(pl.col("day").str.to_date())
            df = df.filter(
                pl.col("day")
                .dt.year()
                .is_between(self.selected_years[0], self.selected_years[1])
            )

            # Add day of week
            df = df.with_columns(pl.col("day").dt.weekday().alias("day_of_week"))

            df = df.join(
                holidays_df, left_on="day", right_on="holiday_date", how="left"
            ).with_columns(
                pl.col("is_holiday").fill_null(False)
            )

            df = df.with_columns(
                ((pl.col("day").dt.weekday() >= 5) | pl.col("is_holiday")).alias(
                    "is_weekend"
                )
            )

            df = df.join(weather_daily, left_on="day", right_on="date", how="left")

            df = df.with_columns(((pl.col("day").dt.month() % 12) // 3).alias("season"))

            consumption_cols = [col for col in df.columns if col.startswith("hh")]
            df = df.with_columns(
                pl.concat_list(consumption_cols).alias("hh_consumption")
            )

            df = df.select(
                [
                    "day_of_week",
                    "is_weekend",
                    "avg_temperature",
                    "avg_humidity",
                    "season",
                    "hh_consumption",
                ]
            )
            result_frames.append(df)

        if result_frames:
            return pl.concat(result_frames)
        else:
            return pl.DataFrame(
                schema={
                    "day_of_week": pl.Int32,
                    "is_weekend": pl.Boolean,
                    "avg_temperature": pl.Float64,
                    "avg_humidity": pl.Float64,
                    "season": pl.Int32,
                    "hh_consumption": pl.List(pl.Float64),
                }
            )

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
                cache_path = os.path.join(self.processor_cache_dir, cache_key)

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
    ) -> Tuple[DataLoader, np.ndarray]:
        """
        Create a DataLoader for test/evaluation data.

        Args:
            data: Input data to evaluate
            batch_size: Batch size for evaluation
            num_workers: Number of worker processes for data loading

        Returns:
            Tuple of (DataLoader for test data, processed_data)
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    acorn_data = AcornData(acorn_group="Comfortable", selected_years=(2011, 2012))
    data = acorn_data.get_data()
    print(data)
