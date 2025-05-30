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
        # Extract year from the day column (assuming format YYYY-MM-DD)
        df = df.with_columns(pl.col("day").str.slice(0, 4).alias("year"))

        # Create hourly data by combining half-hour intervals
        hourly_cols = [
            (pl.col(f"hh_{hour * 2}") + pl.col(f"hh_{hour * 2 + 1}")).alias(f"h_{hour}")
            for hour in range(24)
        ]

        # Add year and hourly columns to dataframe
        df_with_hourly = df.select(pl.col("year"), *hourly_cols)

        # Melt the data to have hour as a column
        melted_df = df_with_hourly.unpivot(
            index=["year"],
            on=[f"h_{hour}" for hour in range(24)],
            variable_name="hour",
            value_name="energy",
        )

        # Extract hour number from column name
        melted_df = melted_df.with_columns(
            pl.col("hour").str.replace("h_", "").cast(pl.Int32)
        )

        # Group by year and hour, then calculate requested statistics
        result = (
            melted_df.group_by(["year", "hour"])
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
        )

        return result

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


class DataStorage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.processor = DataProcessor()
        self.hourly_patterns = None

    def get_hourly_patterns(
        self, years: List[int] | None = None, cols: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Retrieve hourly patterns from the data processor.

        Args:
            years: Optional list of years to filter by.
            cols: Optional list of columns to select.

        Returns:
            DataFrame with hourly patterns.
        """
        if self.hourly_patterns is None:
            data = self.processor.get_hourly_patterns()
            data = data.with_columns(pl.col("year").cast(pl.Int32))
            self.hourly_patterns = data

        result = self.hourly_patterns

        if years is not None:
            result = result.filter(pl.col("year").is_in(years))

        if cols is not None:
            result = result.select(cols)

        return result


storage = DataStorage()


if __name__ == "__main__":
    storage = DataStorage()
    print(storage.get_hourly_patterns())
    print(storage.get_hourly_patterns(years=[2010, 2011], cols=["year", "hour", "energy_mean"]))

