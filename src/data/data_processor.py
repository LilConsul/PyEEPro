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
        Convert half-hourly energy consumption data to hourly patterns and calculate statistics.

        This method transforms raw half-hourly readings (48 readings per day in columns hh_0 to hh_47)
        into hourly aggregated data. It calculates various statistical measures for each hour
        across different years, enabling analysis of hourly consumption patterns.

        Args:
            df: Polars DataFrame with columns:
               - LCLid: Customer identifier
               - day: Date string in format YYYY-MM-DD
               - hh_0 through hh_47: Half-hourly energy readings (48 columns)

        Returns:
            Polars DataFrame with columns:
               - year: Calendar year extracted from dates
               - hour: Hour of day (0-23)
               - energy_median: Median energy consumption for that hour
               - energy_mean: Mean energy consumption for that hour
               - energy_max: Maximum energy consumption for that hour
               - energy_count: Number of data points used in calculation
               - energy_std: Standard deviation of energy consumption
               - energy_sum: Total energy consumption for that hour
               - energy_min: Minimum energy consumption for that hour

        Notes:
            - Missing values in half-hourly readings are filled with zeros
            - Hours are calculated by summing consecutive half-hour readings
            - Results are sorted by year and hour
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

    def get_hourly_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract hourly patterns throughout the day.

        This method serves as a pipeline that:
        1. Loads raw half-hourly energy data from the configured directory (HHBLOCKS_DIR)
        2. Processes the data to extract hourly consumption patterns
        3. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing hourly energy consumption statistics with columns:
            - year: Calendar year extracted from dates (int)
            - hour: Hour of day (0-23) (int)
            - energy_median: Median energy consumption for that hour (float)
            - energy_mean: Mean energy consumption for that hour (float)
            - energy_max: Maximum energy consumption for that hour (float)
            - energy_count: Number of data points used in calculation (int)
            - energy_std: Standard deviation of energy consumption (float)
            - energy_sum: Total energy consumption for that hour (float)
            - energy_min: Minimum energy consumption for that hour (float)

        Raises:
            ValueError: If no valid CSV files are found in the configured directory
        """
        data = self._load_data_from_dir(settings.HHBLOCKS_DIR)
        hourly_patterns = self._process_hourly_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(hourly_patterns)

        return hourly_patterns

    @staticmethod
    def _process_daily_patterns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Process daily energy consumption data to extract patterns by day of week.

        This method analyzes daily energy consumption data and aggregates it by weekday,
        allowing for identification of consumption patterns specific to each day of the week.
        It converts date strings to date objects, extracts weekday information, and
        calculates various statistical metrics for each weekday.

        Args:
            df: Polars DataFrame containing at least:
               - day: Date string in format YYYY-MM-DD
               - energy_median: Median energy consumption for each day
               - energy_mean: Mean energy consumption for each day
               - energy_max: Maximum energy consumption for each day
               - energy_count: Number of data points in each day
               - energy_std: Standard deviation of energy consumption
               - energy_sum: Total energy consumption for each day
               - energy_min: Minimum energy consumption for each day

        Returns:
            Polars DataFrame with columns:
               - year: Calendar year of the data
               - weekday: Day of week as integer (1-7, Monday-Sunday)
               - weekday_name: Name of day of week (Monday through Sunday)
               - energy_median: Average of daily median energy values
               - energy_mean: Average of daily mean energy values
               - energy_max: Maximum energy consumption for that weekday
               - energy_count: Sum of daily count values
               - energy_std: Average of daily standard deviation values
               - energy_sum: Total energy consumption for that weekday
               - energy_min: Minimum energy consumption for that weekday
               - days_count: Number of days included in each group

        Notes:
            - Data is grouped by year and day of week
            - Results are sorted by year and weekday
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

    def get_daily_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by day of week.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Processes the data to extract daily consumption patterns by weekday
        3. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing daily energy consumption statistics with columns:
            - year: Calendar year of the data (int)
            - weekday: Day of week as integer (1-7, Monday-Sunday) (int)
            - weekday_name: Name of day of week ("Monday" through "Sunday") (str)
            - energy_median: Average of daily median energy values (float)
            - energy_mean: Average of daily mean energy values (float)
            - energy_max: Maximum energy consumption for that weekday (float)
            - energy_count: Sum of daily count values (int)
            - energy_std: Average of daily standard deviation values (float)
            - energy_sum: Total energy consumption for that weekday (float)
            - energy_min: Minimum energy consumption for that weekday (float)
            - days_count: Number of days included in each group (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directory
        """
        data = self._load_data_from_dir(settings.DAILYBLOCKS_DIR)
        daily_patterns = self._process_daily_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(daily_patterns)

        return daily_patterns

    @staticmethod
    def _process_weekly_patterns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Process daily energy consumption data to extract patterns by week of year.

        This method aggregates daily energy consumption data by week of year,
        enabling analysis of seasonal patterns throughout the year. It converts
        date strings to date objects, extracts week number information, and
        calculates various statistical metrics for each week of the year.

        Args:
            df: Polars DataFrame containing at least:
               - day: Date string in format YYYY-MM-DD
               - energy_median: Median energy consumption for each day
               - energy_mean: Mean energy consumption for each day
               - energy_max: Maximum energy consumption for each day
               - energy_count: Number of data points in each day
               - energy_std: Standard deviation of energy consumption
               - energy_sum: Total energy consumption for each day
               - energy_min: Minimum energy consumption for each day

        Returns:
            Polars DataFrame with columns:
               - year: Calendar year of the data
               - week: Week number within the year (1-53)
               - energy_median: Average of daily median energy values for that week
               - energy_mean: Average of daily mean energy values for that week
               - energy_max: Maximum energy consumption for that week
               - energy_count: Sum of daily count values for that week
               - energy_std: Average of daily standard deviation values for that week
               - energy_sum: Total energy consumption for that week
               - energy_min: Minimum energy consumption for that week
               - weeks_count: Number of days included in each weekly group

        Notes:
            - Data is grouped by year and week number
            - Results are sorted by year and week number
            - Week numbering follows ISO standard (weeks start on Monday)
        """
        # Use lazy evaluation for query optimization
        lazy_df = df.lazy()

        # Extract year and week of year from the day column
        result = (
            lazy_df.with_columns(
                [
                    pl.col("day").str.to_date().dt.year().alias("year"),
                    pl.col("day").str.to_date().dt.week().alias("week"),
                ]
            )
            .group_by(["year", "week"])
            .agg(
                energy_median=pl.col("energy_median").mean(),
                energy_mean=pl.col("energy_mean").mean(),
                energy_max=pl.col("energy_max").max(),
                energy_count=pl.col("energy_count").sum(),
                energy_std=pl.col("energy_std").mean(),
                energy_sum=pl.col("energy_sum").sum(),
                energy_min=pl.col("energy_min").min(),
                weeks_count=pl.count(),
            )
            .sort(["year", "week"])
            .collect()
        )

        return result

    def get_weekly_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by week of year.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Processes the data to extract weekly consumption patterns
        3. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing weekly energy consumption statistics with columns:
            - year: Calendar year of the data (int)
            - week: Week number within the year (1-53) (int)
            - energy_median: Average of daily median energy values for that week (float)
            - energy_mean: Average of daily mean energy values for that week (float)
            - energy_max: Maximum energy consumption for that week (float)
            - energy_count: Sum of daily count values for that week (int)
            - energy_std: Average of daily standard deviation values for that week (float)
            - energy_sum: Total energy consumption for that week (float)
            - energy_min: Minimum energy consumption for that week (float)
            - weeks_count: Number of days included in each weekly group (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directory
        """
        data = self._load_data_from_dir(settings.DAILYBLOCKS_DIR)
        weekly_patterns = self._process_weekly_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(weekly_patterns)

        return weekly_patterns

    @staticmethod
    def _process_seconal_patterns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by season of year.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Processes the data to extract seasonal consumption patterns
        3. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing seasonal energy consumption statistics with columns:
            - year: Calendar year of the data (int)
            - season: Season name (Winter, Spring, Summer, Fall) (str)
            - week: Week number within the season (1-12) (int)
            - energy_median: Average of daily median energy values for that season (float)
            - energy_mean: Average of daily mean energy values for that season (float)
            - energy_max: Maximum energy consumption for that season (float)
            - energy_count: Sum of daily count values for that season (int)
            - energy_std: Average of daily standard deviation values for that season (float)
            - energy_sum: Total energy consumption for that season (float)
            - energy_min: Minimum energy consumption for that season (float)
            - days_count: Number of days included in each season group (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directory
        """
        # Use lazy evaluation for query optimization
        lazy_df = df.lazy()

        result = (
            lazy_df
            # Create year and week columns
            .with_columns(
                [
                    pl.col("day").str.to_date().dt.year().alias("year"),
                    pl.col("day").str.to_date().dt.week().alias("week_num"),
                ]
            )
            # Add season column based on week number
            .with_columns(
                [
                    pl.when((pl.col("week_num") >= 1) & (pl.col("week_num") <= 9))
                    .then(pl.lit("Winter"))
                    .when((pl.col("week_num") >= 49) & (pl.col("week_num") <= 53))
                    .then(pl.lit("Winter"))
                    .when((pl.col("week_num") >= 10) & (pl.col("week_num") <= 22))
                    .then(pl.lit("Spring"))
                    .when((pl.col("week_num") >= 23) & (pl.col("week_num") <= 35))
                    .then(pl.lit("Summer"))
                    .when((pl.col("week_num") >= 36) & (pl.col("week_num") <= 48))
                    .then(pl.lit("Fall"))
                    .alias("season")
                ]
            )
            # Add week column (1-12 within each season)
            .with_columns(
                [
                    pl.when(pl.col("season") == "Winter")
                    .then(
                        pl.when(pl.col("week_num") <= 9)
                        .then(pl.col("week_num"))
                        .otherwise((pl.col("week_num") - 48 + 9) % 12 + 1)
                    )
                    .when(pl.col("season") == "Spring")
                    .then(((pl.col("week_num") - 10) % 12) + 1)
                    .when(pl.col("season") == "Summer")
                    .then(((pl.col("week_num") - 23) % 12) + 1)
                    .when(pl.col("season") == "Fall")
                    .then(((pl.col("week_num") - 36) % 12) + 1)
                    .alias("week")
                ]
            )
            # Group by and aggregate
            .group_by(["year", "season", "week"])
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
            # Add season order for sorting
            .with_columns(
                [
                    pl.when(pl.col("season") == "Winter")
                    .then(1)
                    .when(pl.col("season") == "Spring")
                    .then(2)
                    .when(pl.col("season") == "Summer")
                    .then(3)
                    .when(pl.col("season") == "Fall")
                    .then(4)
                    .alias("season_order")
                ]
            )
            .sort(["year", "season_order", "week"])
            .drop("season_order")
            .collect()
        )

        return result

    def get_seconal_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by season of year.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Processes the data to extract seasonal consumption patterns
        3. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing seasonal energy consumption statistics with columns:
            - year: Calendar year of the data (int)
            - season: Season name (Winter, Spring, Summer, Fall) (str)
            - week: Week number within the season (1-12) (int)
            - energy_median: Average of daily median energy values for that season (float)
            - energy_mean: Average of daily mean energy values for that season (float)
            - energy_max: Maximum energy consumption for that season (float)
            - energy_count: Sum of daily count values for that season (int)
            - energy_std: Average of daily standard deviation values for that season (float)
            - energy_sum: Total energy consumption for that season (float)
            - energy_min: Minimum energy consumption for that season (float)
            - days_count: Number of days included in each season group (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directory
        """
        data = self._load_data_from_dir(settings.DAILYBLOCKS_DIR)
        seasonal_patterns = self._process_seconal_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(seasonal_patterns)

        return seasonal_patterns


