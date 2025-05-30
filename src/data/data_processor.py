import polars as pl
from pathlib import Path
from src.app.config import setting


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
                if setting.DEBUG:
                    print(f"Queued {file.name} for processing")
            except Exception as e:
                print(f"Error queuing {file.name}: {str(e)}")

        if not lazy_frames:
            raise ValueError("No valid CSV files could be loaded")

        return pl.concat(lazy_frames).collect()

    @staticmethod
    def process_half_hourly_to_hourly_patterns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert half-hourly data to hourly and perform comprehensive EDA on hourly consumption patterns.

        Args:
            df: Polars DataFrame with columns LCLid, day, hh_0, hh_1, ..., hh_47

        Returns:
            Polars DataFrame with hourly consumption statistics for EDA
        """
        # Create hourly data by combining half-hour intervals
        hourly_data_expressions = [
            (pl.col(f"hh_{hour * 2}") + pl.col(f"hh_{hour * 2 + 1}")).alias(f"h_{hour}")
            for hour in range(24)
        ]

        # Select the hourly data columns
        hourly_data = df.select(hourly_data_expressions)

        # Compute comprehensive statistics for each hour
        stats = []
        for hour in range(24):
            hour_col = f"h_{hour}"
            hour_stats = {
                "hour": hour,
                "mean": hourly_data[hour_col].mean(),
                "median": hourly_data[hour_col].median(),
                "min": hourly_data[hour_col].min(),
                "max": hourly_data[hour_col].max(),
                "std_dev": hourly_data[hour_col].std(),
                "q25": hourly_data[hour_col].quantile(0.25),
                "q75": hourly_data[hour_col].quantile(0.75),
            }
            stats.append(hour_stats)

        result = pl.DataFrame(stats)

        # Calculate IQR
        result = result.with_columns((pl.col("q75") - pl.col("q25")).alias("iqr"))

        # Calculate percentage of daily consumption
        total_daily_mean = result["mean"].sum()
        result = result.with_columns(
            (pl.col("mean") / total_daily_mean * 100).alias("percent_of_daily")
        )

        # Add hour classification based on consumption level
        mean_values = result["mean"].to_list()
        high_threshold = pl.Series(mean_values).quantile(0.66)
        low_threshold = pl.Series(mean_values).quantile(0.33)

        result = result.with_columns(
            pl.when(pl.col("mean") >= high_threshold)
            .then(pl.lit("Peak"))
            .when(pl.col("mean") <= low_threshold)
            .then(pl.lit("Off-Peak"))
            .otherwise(pl.lit("Mid-Level"))
            .alias("consumption_category")
        )

        # Sort by hour for better readability
        result = result.sort("hour")

        return result

    def get_hourly_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract hourly patterns.

        Returns:
            DataFrame with hourly consumption patterns
        """
        data = self.load_data_from_dir(setting.HHBLOCKS_DIR)
        hourly_patterns = self.process_half_hourly_to_hourly_patterns(data)

        if setting.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(hourly_patterns)

        return hourly_patterns


if __name__ == "__main__":
    processor = DataProcessor()
    patterns = processor.get_hourly_patterns()
