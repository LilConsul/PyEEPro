from typing import List
import polars as pl
from pathlib import Path
from settings import settings
import streamlit as st


class DataProcessor:
    @staticmethod
    def load_data_from_dir(directory: Path) -> pl.DataFrame:
        """
        Load all CSV files from a directory and concatenate them into a single Polars DataFrame.

        Args:
            directory: Path to the directory containing CSV files.

        Returns:
            Polars DataFrame containing all records from the CSV files.
        """
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {directory}")

        # Use scan_csv for lazy loading and concat efficiently
        lazy_frames = []
        for file in csv_files:
            try:
                lazy_df = pl.scan_csv(file)
                lazy_frames.append(lazy_df)
                if settings.DEBUG:
                    print(f"Queued {file.name} for processing")
            except Exception as e:
                print(f"Error queuing {file.name}: {str(e)}")

        if not lazy_frames:
            raise ValueError("No valid CSV files could be loaded")

        return pl.concat(lazy_frames).collect()

    @staticmethod
    def process_hourly_patterns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert half-hourly data to hourly and calculate energy statistics grouped by year.

        Args:
            df: Polars DataFrame with columns LCLid, day, hh_0, hh_1, ..., hh_47

        Returns:
            Polars DataFrame with hourly consumption statistics by year
        """
        # Use lazy evaluation for query optimization
        lazy_df = df.lazy()

        lazy_df = lazy_df.with_columns(
            pl.col("day").str.slice(0, 4).cast(pl.Int32).alias("year")
        )

        hourly_cols = [
            (
                pl.col(f"hh_{hour * 2}").fill_null(0)
                + pl.col(f"hh_{hour * 2 + 1}").fill_null(0)
            ).alias(f"h_{hour}")
            for hour in range(24)
        ]

        result = (
            lazy_df.select("year", *hourly_cols)
            .unpivot(
                index=["year"],
                on=[f"h_{hour}" for hour in range(24)],
                variable_name="hour",
                value_name="energy",
            )
            .with_columns(pl.col("hour").str.replace("h_", "").cast(pl.Int32))
            .filter(pl.col("energy").is_not_null())
            .group_by(["year", "hour"])
            .agg(
                energy_median=pl.col("energy").median(),
                energy_mean=pl.col("energy").mean(),
                energy_max=pl.col("energy").max(),
                energy_count=pl.col("energy").count(),
                energy_std=pl.col("energy").std(),
                energy_sum=pl.col("energy").sum(),
                energy_min=pl.col("energy").min(),
            )
            .sort(["year", "hour"])
            .collect()
        )

        return result

    @staticmethod
    def process_daily_patterns(df: pl.DataFrame) -> pl.DataFrame:
        return df

    def get_hourly_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract hourly patterns.

        Returns:
            DataFrame with hourly consumption patterns
        """
        data = self.load_data_from_dir(settings.HHBLOCKS_DIR)
        hourly_patterns = self.process_hourly_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(hourly_patterns)

        return hourly_patterns

    def get_dyily_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract daily patterns.

        Returns:
            DataFrame with daily consumption patterns
        """
        data = self.load_data_from_dir(settings.DYLYBLOCKS_DIR)
        daily_patterns = self.process_daily_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=20, tbl_cols=-1):
                print(daily_patterns)

        return daily_patterns


class DataStorage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.processor = DataProcessor()
        settings.CACHE_DIR.mkdir(exist_ok=True)
        self.hourly_patterns = self._load_cache("hourly_patterns.csv")

    @staticmethod
    def _load_cache(filename: str) -> pl.DataFrame | None:
        """Load data from cache if it exists."""
        cache_path = settings.CACHE_DIR / filename
        if cache_path.exists():
            try:
                return pl.read_csv(cache_path)
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error loading cache {filename}: {str(e)}")
        return None

    @staticmethod
    def _save_cache(data: pl.DataFrame, filename: str) -> None:
        """Save processed data to cache."""
        cache_path = settings.CACHE_DIR / filename
        try:
            data.write_csv(cache_path)
            if settings.DEBUG:
                print(f"Cached data saved to {filename}")
        except Exception as e:
            if settings.DEBUG:
                print(f"Error saving cache {filename}: {str(e)}")

    @staticmethod
    def remove_cache(filename: str) -> None:
        """Remove a specific cache file."""
        cache_path = settings.CACHE_DIR / filename
        if cache_path.exists():
            try:
                cache_path.unlink()
                if settings.DEBUG:
                    print(f"Cache {filename} removed")
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error removing cache {filename}: {str(e)}")
        else:
            if settings.DEBUG:
                print(f"Cache {filename} does not exist")

    @staticmethod
    def remove_all_caches() -> None:
        """Remove all cache files."""
        if settings.CACHE_DIR.exists():
            try:
                for file in settings.CACHE_DIR.glob("*.csv"):
                    file.unlink()
                if settings.DEBUG:
                    print("All caches removed")
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error removing caches: {str(e)}")
        else:
            if settings.DEBUG:
                print("Cache directory does not exist")

    @staticmethod
    def get_cached_files() -> List[str]:
        """Get a list of all cached files."""
        if settings.CACHE_DIR.exists():
            return [file.name for file in settings.CACHE_DIR.glob("*.csv")]
        return []

    def get_hourly_patterns(
        self, years: List[int] | None = None, cols: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Retrieve hourly patterns from cache or process if not available.

        Args:
            years: Optional list of years to filter by.
            cols: Optional list of columns to select.

        Returns:
            DataFrame with hourly patterns.
        """
        if self.hourly_patterns is None:
            data = self.processor.get_hourly_patterns()
            self.hourly_patterns = data
            self._save_cache(data, "hourly_patterns.csv")

        result = self.hourly_patterns

        if years is not None:
            result = result.filter(pl.col("year").is_in(years))

        if cols is not None:
            result = result.select(cols)

        return result


storage = DataStorage()


if __name__ == "__main__":
    storage = DataStorage()
    print(storage.get_cached_files())

