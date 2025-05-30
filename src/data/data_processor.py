import polars as pl
from pathlib import Path
from settings import settings


class DataProcessor:
    @staticmethod
    def _load_data_from_dir(directory: Path) -> pl.DataFrame:
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
    def _process_hourly_patterns(df: pl.DataFrame) -> pl.DataFrame:
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
    def _process_daily_patterns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Process daily energy consumption data to extract patterns by day of week.

        Args:
            df: Polars DataFrame with daily energy data including 'day' column
                and energy statistics

        Returns:
            Polars DataFrame with consumption statistics grouped by year and day of week
        """
        # Use lazy evaluation for query optimization
        lazy_df = df.lazy()

        # Extract year and day of week from the day column
        result = (
            lazy_df.with_columns(
                [
                    pl.col("day").str.to_date().dt.year().alias("year"),
                    pl.col("day").str.to_date().dt.weekday().alias("weekday"),
                ]
            )
            # Add weekday name mapping
            .with_columns(
                pl.when(pl.col("weekday") == 1)
                .then(pl.lit("Monday"))
                .when(pl.col("weekday") == 2)
                .then(pl.lit("Tuesday"))
                .when(pl.col("weekday") == 3)
                .then(pl.lit("Wednesday"))
                .when(pl.col("weekday") == 4)
                .then(pl.lit("Thursday"))
                .when(pl.col("weekday") == 5)
                .then(pl.lit("Friday"))
                .when(pl.col("weekday") == 6)
                .then(pl.lit("Saturday"))
                .when(pl.col("weekday") == 7)
                .then(pl.lit("Sunday"))
                .alias("weekday_name")
            )
            .group_by(["year", "weekday", "weekday_name"])
            .agg(
                energy_median=pl.col("energy_median").mean(),
                energy_mean=pl.col("energy_mean").mean(),
                energy_max=pl.col("energy_max").max(),
                energy_count=pl.col("energy_count").sum(),
                energy_std=pl.col("energy_std").mean(),
                energy_sum=pl.col("energy_sum").sum(),
                energy_min=pl.col("energy_min").min(),
                days_count=pl.count(),
            )
            .sort(["year", "weekday"])
            .collect()
        )

        return result

    def get_hourly_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract hourly patterns.

        Returns:
            DataFrame with hourly consumption patterns
        """
        data = self._load_data_from_dir(settings.HHBLOCKS_DIR)
        hourly_patterns = self._process_hourly_patterns(data)

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
        data = self._load_data_from_dir(settings.DYLYBLOCKS_DIR)
        daily_patterns = self._process_daily_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(daily_patterns)

        return daily_patterns


