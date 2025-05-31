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
    def _load_data_from_file(file: Path) -> pl.DataFrame:
        """
        Load a single CSV file into a Polars DataFrame.

        Args:
            file: Path to the CSV file.

        Returns:
            Polars DataFrame containing the records from the CSV file.
        """
        try:
            df = pl.read_csv(file)
            if settings.DEBUG:
                print(f"Loaded data from {file.name}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from {file}: {str(e)}")

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
    def _process_seasonal_patterns(df: pl.DataFrame) -> pl.DataFrame:
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
                    # Define seasons based on week numbers
                    pl.when((pl.col("week_num") >= 1) & (pl.col("week_num") <= 9))
                    .then(pl.lit("Winter"))
                    .when((pl.col("week_num") >= 10) & (pl.col("week_num") <= 22))
                    .then(pl.lit("Spring"))
                    .when((pl.col("week_num") >= 23) & (pl.col("week_num") <= 35))
                    .then(pl.lit("Summer"))
                    .when((pl.col("week_num") >= 36) & (pl.col("week_num") <= 48))
                    .then(pl.lit("Fall"))
                    .when((pl.col("week_num") >= 49) & (pl.col("week_num") <= 53))
                    .then(pl.lit("Winter"))
                    .otherwise(pl.lit("Unknown"))
                    .alias("season"),
                    # Create a season_sequence field to order seasons chronologically within a year
                    pl.when((pl.col("week_num") >= 1) & (pl.col("week_num") <= 9))
                    .then(1)  # Early year winter comes first
                    .when((pl.col("week_num") >= 10) & (pl.col("week_num") <= 22))
                    .then(2)  # Spring comes second
                    .when((pl.col("week_num") >= 23) & (pl.col("week_num") <= 35))
                    .then(3)  # Summer comes third
                    .when((pl.col("week_num") >= 36) & (pl.col("week_num") <= 48))
                    .then(4)  # Fall comes fourth
                    .when((pl.col("week_num") >= 49) & (pl.col("week_num") <= 53))
                    .then(5)  # Late year winter comes last
                    .alias("season_sequence"),
                ]
            )
            # Add simplified week column within each season - fix the winter week numbering
            .with_columns(
                [
                    pl.when(pl.col("season") == "Winter")
                    .then(
                        pl.when(pl.col("week_num") >= 49)
                        .then(pl.col("week_num") - 48)  # Weeks 49-53 become 1-5
                        .otherwise(pl.col("week_num") + 4)  # Weeks 1-9 become 5-13
                    )
                    .when(pl.col("season") == "Spring")
                    .then(pl.col("week_num") - 9)  # Weeks 10-22 become 1-13
                    .when(pl.col("season") == "Summer")
                    .then(pl.col("week_num") - 22)  # Weeks 23-35 become 1-13
                    .when(pl.col("season") == "Fall")
                    .then(pl.col("week_num") - 35)  # Weeks 36-48 become 1-13
                    .alias("week")
                ]
            )
            # Group by and aggregate
            .group_by(["year", "season", "season_sequence", "week"])
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
            # Sort by year and season_sequence to get the natural flow of seasons
            .sort(["year", "season_sequence", "week"])
            .drop("season_sequence")  # Remove helper column
            .collect()
        )

        return result

    def get_seasonal_patterns(self) -> pl.DataFrame:
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
        seasonal_patterns = self._process_seasonal_patterns(data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(seasonal_patterns)

        return seasonal_patterns

    @staticmethod
    def _process_weekday_vs_weekend_patterns(
        df: pl.DataFrame, holidays_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Process daily energy consumption data to extract patterns by weekday vs weekend.
        Bank holidays are treated as weekend days regardless of the actual day of week.

        This method analyzes daily energy consumption data and aggregates it by weekday/weekend status,
        allowing for identification of consumption patterns specific to weekdays versus weekends.

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
               - is_weekend: Boolean flag (True for weekend days, False for weekdays)
               - energy_median: Average of daily median energy values
               - energy_mean: Average of daily mean energy values
               - energy_max: Maximum energy consumption
               - energy_count: Sum of daily count values
               - energy_std: Average of daily standard deviation values
               - energy_sum: Total energy consumption
               - energy_min: Minimum energy consumption
               - days_count: Number of days included in each group
        """
        # Use lazy evaluation for query optimization
        lazy_df = df.lazy()
        holidays_lazy = holidays_df.lazy()

        # Prepare holidays list - convert to date format
        holidays_list = (
            holidays_lazy.select(
                pl.col("Bank holidays").str.to_date().alias("holiday_date")
            )
            .collect()
            .get_column("holiday_date")
            .to_list()
        )

        # Extract year and determine if day is weekend
        result = (
            lazy_df.with_columns(
                [
                    pl.col("day").str.to_date().alias("date"),
                    pl.col("day").str.to_date().dt.year().alias("year"),
                    pl.col("day").str.to_date().dt.weekday().alias("weekday"),
                ]
            )
            # Add is_weekend flag (weekday in Polars: 1=Monday, ..., 7=Sunday)
            # Also mark holidays as weekend
            .with_columns((pl.col("weekday") >= 6).alias("is_regular_weekend"))
            .with_columns(
                (
                    pl.col("is_regular_weekend") | pl.col("date").is_in(holidays_list)
                ).alias("is_weekend")
            )
            .group_by(["year", "is_weekend"])
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
            .sort(["year", "is_weekend"])
            .collect()
        )

        return result

    def get_weekday_vs_weekend_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by weekday vs weekend.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Processes the data to differentiate between weekday and weekend consumption patterns
        3. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing weekday/weekend energy consumption statistics with columns:
            - year: Calendar year of the data (int)
            - is_weekend: Boolean flag (True for weekend days, False for weekdays) (bool)
            - energy_median: Average of daily median energy values (float)
            - energy_mean: Average of daily mean energy values (float)
            - energy_max: Maximum energy consumption (float)
            - energy_count: Sum of daily count values (int)
            - energy_std: Average of daily standard deviation values (float)
            - energy_sum: Total energy consumption (float)
            - energy_min: Minimum energy consumption (float)
            - days_count: Number of days included in each group (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directory
        """
        data = self._load_data_from_dir(settings.DAILYBLOCKS_DIR)
        holidays = self._load_data_from_file(settings.HOLIDAYS_FILE)
        weekend_patterns = self._process_weekday_vs_weekend_patterns(data, holidays)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(weekend_patterns)

        return weekend_patterns

    @staticmethod
    def _process_household_patterns(
        df: pl.DataFrame, household_info: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by household type.

        This method joins energy consumption data with household metadata and aggregates
        it by tariff type (Standard/ToU) and socioeconomic categories (Acorn groups).
        It calculates various statistical metrics for each household segment.

        Args:
            df: Polars DataFrame containing energy consumption data with columns:
                - LCLid: Customer identifier
                - day: Date string in format YYYY-MM-DD
                - energy_median: Median energy consumption for each day
                - energy_mean: Mean energy consumption for each day
                - energy_max: Maximum energy consumption for each day
                - energy_count: Number of data points in each day
                - energy_std: Standard deviation of energy consumption
                - energy_sum: Total energy consumption for each day
                - energy_min: Minimum energy consumption for each day

            household_info: Polars DataFrame containing household metadata with columns:
                - LCLid: Customer identifier
                - stdorToU: Tariff type (Standard or Time of Use)
                - Acorn: Detailed socioeconomic category
                - Acorn_grouped: Simplified socioeconomic group (Affluent, Comfortable, etc.)

        Returns:
            Polars DataFrame with columns:
                - stdorToU: Tariff type (Standard or Time of Use) (str)
                - Acorn_grouped: Simplified socioeconomic group (str)
                - Acorn: Detailed socioeconomic category (str)
                - energy_median: Average of daily median energy values (float)
                - energy_mean: Average of daily mean energy values (float)
                - energy_max: Maximum energy consumption (float)
                - energy_count: Sum of daily count values (int)
                - energy_std: Average of daily standard deviation values (float)
                - energy_sum: Total energy consumption (float)
                - energy_min: Minimum energy consumption (float)
                - household_count: Number of unique households in each group (int)
                - days_count: Number of days included in each group (int)

        Notes:
            - Results are sorted by Acorn_grouped, Acorn, and tariff type
            - Missing values in the join are excluded (inner join)
        """
        # Use lazy evaluation for query optimization
        lazy_df = df.lazy()
        lazy_household_info = household_info.lazy()

        # Extract year from day column
        lazy_df = lazy_df.with_columns(
            pl.col("day").str.to_date().dt.year().alias("year")
        )

        # Join the energy data with household information
        joined_df = lazy_df.join(
            lazy_household_info.select("LCLid", "stdorToU", "Acorn", "Acorn_grouped"),
            on="LCLid",
            how="inner",
        )

        # Group by year, tariff type, and Acorn group
        result = (
            joined_df.group_by(["stdorToU", "Acorn_grouped", "Acorn"])
            .agg(
                energy_median=pl.col("energy_median").mean(),
                energy_mean=pl.col("energy_mean").mean(),
                energy_max=pl.col("energy_max").max(),
                energy_count=pl.col("energy_count").sum(),
                energy_std=pl.col("energy_std").mean(),
                energy_sum=pl.col("energy_sum").sum(),
                energy_min=pl.col("energy_min").min(),
                household_count=pl.col("LCLid").n_unique(),
                days_count=pl.count(),
            )
            .sort(["Acorn_grouped", "Acorn", "stdorToU"])
            .collect()
        )

        return result

    def get_household_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to extract patterns by household type.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Loads household metadata from the configured file (INFORMATION_HOUSEHOLD_FILE)
        3. Processes the data to extract consumption patterns by socioeconomic group and tariff type
        4. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing household energy consumption statistics with columns:
            - stdorToU: Tariff type (Standard or Time of Use) (str)
            - Acorn_grouped: Simplified socioeconomic group (Affluent, Comfortable, etc.) (str)
            - Acorn: Detailed socioeconomic category (str)
            - energy_median: Average of daily median energy values (float)
            - energy_mean: Average of daily mean energy values (float)
            - energy_max: Maximum energy consumption (float)
            - energy_count: Sum of daily count values (int)
            - energy_std: Average of daily standard deviation values (float)
            - energy_sum: Total energy consumption (float)
            - energy_min: Minimum energy consumption (float)
            - household_count: Number of unique households in each group (int)
            - days_count: Number of days included in each group (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directories or files
        """
        data = self._load_data_from_dir(settings.DAILYBLOCKS_DIR)
        household_info = self._load_data_from_file(settings.INFORMATION_HOUSEHOLD_FILE)
        household_patterns = self._process_household_patterns(data, household_info)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(household_patterns)

        return household_patterns

    @staticmethod
    def _process_temperature_energy_patterns(
        energy_df: pl.DataFrame, weather_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Process energy consumption data and correlate it with temperature data.

        This method joins daily energy consumption data with weather data to analyze
        how temperature affects energy usage patterns throughout the year.

        Args:
            energy_df: Polars DataFrame containing energy consumption data with columns:
                - day: Date string in format YYYY-MM-DD
                - energy_median: Median energy consumption for each day
                - energy_mean: Mean energy consumption for each day
                - energy_max: Maximum energy consumption for each day
                - energy_count: Number of data points in each day
                - energy_std: Standard deviation of energy consumption
                - energy_sum: Total energy consumption for each day
                - energy_min: Minimum energy consumption for each day

            weather_df: Polars DataFrame containing weather data with columns:
                - date: Date in datetime format
                - temperatureMin: Minimum temperature of the day in Celsius
                - temperatureMax: Maximum temperature of the day in Celsius
                - temperatureMean: Mean temperature of the day (calculated)

        Returns:
            Polars DataFrame with columns:
                - year: Calendar year of the data (int)
                - month: Month number (1-12) (int)
                - month_name: Name of month (str)
                - temp_bin: Temperature range bin (str)
                - avg_temperature: Average temperature in that bin (float)
                - energy_median: Average of daily median energy values (float)
                - energy_mean: Average of daily mean energy values (float)
                - energy_max: Maximum energy consumption (float)
                - energy_count: Sum of daily count values (int)
                - energy_std: Average of daily standard deviation values (float)
                - energy_sum: Total energy consumption (float)
                - energy_min: Minimum energy consumption (float)
                - days_count: Number of days included in each temperature bin (int)
        """
        # Use lazy evaluation for query optimization
        lazy_energy_df = energy_df.lazy()
        lazy_weather_df = weather_df.lazy()

        # Prepare energy data - convert day to date format
        prepared_energy = lazy_energy_df.with_columns(
            [
                pl.col("day").str.to_date().alias("date"),
                pl.col("day").str.to_date().dt.year().alias("year"),
                pl.col("day").str.to_date().dt.month().alias("month"),
            ]
        )

        # Prepare weather data - convert time to date and calculate mean temperature
        prepared_weather = lazy_weather_df.with_columns(
            [
                pl.col("time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .dt.date()
                .alias("date"),
                ((pl.col("temperatureMax") + pl.col("temperatureMin")) / 2).alias(
                    "temperatureMean"
                )
            ]
        )

        # Join energy and weather data on date
        joined_df = prepared_energy.join(
            prepared_weather.select(
                "date", "temperatureMin", "temperatureMax", "temperatureMean"
            ),
            on="date",
            how="inner",
        )

        # Create temperature bins
        joined_df = joined_df.with_columns(
            [
                pl.when(pl.col("temperatureMean") < 0)
                .then(pl.lit("Below 0°C"))
                .when(pl.col("temperatureMean") < 5)
                .then(pl.lit("0-5°C"))
                .when(pl.col("temperatureMean") < 10)
                .then(pl.lit("5-10°C"))
                .when(pl.col("temperatureMean") < 15)
                .then(pl.lit("10-15°C"))
                .when(pl.col("temperatureMean") < 20)
                .then(pl.lit("15-20°C"))
                .when(pl.col("temperatureMean") < 25)
                .then(pl.lit("20-25°C"))
                .otherwise(pl.lit("Above 25°C"))
                .alias("temp_bin"),
                pl.when(pl.col("month") == 1)
                .then(pl.lit("January"))
                .when(pl.col("month") == 2)
                .then(pl.lit("February"))
                .when(pl.col("month") == 3)
                .then(pl.lit("March"))
                .when(pl.col("month") == 4)
                .then(pl.lit("April"))
                .when(pl.col("month") == 5)
                .then(pl.lit("May"))
                .when(pl.col("month") == 6)
                .then(pl.lit("June"))
                .when(pl.col("month") == 7)
                .then(pl.lit("July"))
                .when(pl.col("month") == 8)
                .then(pl.lit("August"))
                .when(pl.col("month") == 9)
                .then(pl.lit("September"))
                .when(pl.col("month") == 10)
                .then(pl.lit("October"))
                .when(pl.col("month") == 11)
                .then(pl.lit("November"))
                .when(pl.col("month") == 12)
                .then(pl.lit("December"))
                .alias("month_name"),
            ]
        )

        # Group by year, month and temperature bin
        result = (
            joined_df.group_by(["year", "month", "month_name", "temp_bin"])
            .agg(
                avg_temperature=pl.col("temperatureMean").mean(),
                energy_median=pl.col("energy_median").mean(),
                energy_mean=pl.col("energy_mean").mean(),
                energy_max=pl.col("energy_max").max(),
                energy_count=pl.col("energy_count").sum(),
                energy_std=pl.col("energy_std").mean(),
                energy_sum=pl.col("energy_sum").sum(),
                energy_min=pl.col("energy_min").min(),
                days_count=pl.count(),
            )
            .sort(["year", "month", "temp_bin"])
            .collect()
        )

        return result

    @staticmethod
    def _process_temperature_hourly_energy(
        energy_df: pl.DataFrame, weather_hourly_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Process hourly energy consumption data and correlate it with hourly temperature data.

        This method joins half-hourly energy consumption data with hourly weather data to analyze
        how temperature affects energy usage patterns throughout the day.

        Args:
            energy_df: Polars DataFrame containing half-hourly energy consumption data with columns:
                - LCLid: Customer identifier
                - day: Date string in format YYYY-MM-DD
                - hh_0 through hh_47: Half-hourly energy readings (48 columns)

            weather_hourly_df: Polars DataFrame containing hourly weather data with columns:
                - time: Time string in format YYYY-MM-DD HH:MM:SS
                - temperature: Hourly temperature in Celsius
                - humidity: Hourly humidity

        Returns:
            Polars DataFrame with columns:
                - hour: Hour of day (0-23) (int)
                - temp_bin: Temperature range bin (str)
                - avg_temperature: Average temperature in that bin (float)
                - avg_humidity: Average humidity in that bin (float)
                - energy: Average energy consumption (float)
                - count: Number of data points in each bin (int)
        """
        # Use lazy evaluation for query optimization
        lazy_energy_df = energy_df.lazy()
        lazy_weather_df = weather_hourly_df.lazy()

        # Extract date and hour from weather data
        weather_prep = lazy_weather_df.with_columns(
            [
                pl.col("time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .dt.date()
                .alias("date"),
                pl.col("time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .dt.hour()
                .alias("hour"),
            ]
        )

        # Convert half-hourly energy data to hourly (by summing consecutive half-hours)
        hourly_cols = [
            (
                pl.col(f"hh_{hour * 2}").fill_null(0)
                + pl.col(f"hh_{hour * 2 + 1}").fill_null(0)
            ).alias(f"h_{hour}")
            for hour in range(24)
        ]

        energy_prep = (
            lazy_energy_df.with_columns(pl.col("day").str.to_date().alias("date"))
            .select("date", *hourly_cols)
            .melt(
                id_vars=["date"],
                value_vars=[f"h_{hour}" for hour in range(24)],
                variable_name="hour_str",
                value_name="energy",
            )
            .with_columns(
                pl.col("hour_str").str.replace("h_", "").cast(pl.Int32).alias("hour")
            )
            .drop("hour_str")
        )

        # Join energy and weather data on date and hour
        joined_df = energy_prep.join(
            weather_prep.select("date", "hour", "temperature", "humidity"),
            on=["date", "hour"],
            how="inner",
        )

        # Create temperature bins
        joined_df = joined_df.with_columns(
            pl.when(pl.col("temperature") < 0)
            .then(pl.lit("Below 0°C"))
            .when(pl.col("temperature") < 5)
            .then(pl.lit("0-5°C"))
            .when(pl.col("temperature") < 10)
            .then(pl.lit("5-10°C"))
            .when(pl.col("temperature") < 15)
            .then(pl.lit("10-15°C"))
            .when(pl.col("temperature") < 20)
            .then(pl.lit("15-20°C"))
            .when(pl.col("temperature") < 25)
            .then(pl.lit("20-25°C"))
            .otherwise(pl.lit("Above 25°C"))
            .alias("temp_bin")
        )

        # Group by hour and temperature bin
        result = (
            joined_df.group_by(["hour", "temp_bin"])
            .agg(
                avg_temperature=pl.col("temperature").mean(),
                avg_humidity=pl.col("humidity").mean(),
                energy=pl.col("energy").mean(),
                count=pl.count(),
            )
            .sort(["hour", "temp_bin"])
            .collect()
        )

        return result

    def get_temperature_energy_patterns(self) -> pl.DataFrame:
        """
        Process energy consumption data to analyze how it correlates with temperature.

        This method serves as a pipeline that:
        1. Loads daily energy data from the configured directory (DAILYBLOCKS_DIR)
        2. Loads daily weather data from the configured file (WEATHER_DAILY_FILE)
        3. Processes the data to extract energy consumption patterns by temperature ranges
        4. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing temperature-energy statistics with columns:
            - year: Calendar year of the data (int)
            - month: Month number (1-12) (int)
            - month_name: Name of month (str)
            - temp_bin: Temperature range bin (str)
            - avg_temperature: Average temperature in that bin (float)
            - energy_median: Average of daily median energy values (float)
            - energy_mean: Average of daily mean energy values (float)
            - energy_max: Maximum energy consumption (float)
            - energy_count: Sum of daily count values (int)
            - energy_std: Average of daily standard deviation values (float)
            - energy_sum: Total energy consumption (float)
            - energy_min: Minimum energy consumption (float)
            - days_count: Number of days included in each temperature bin (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directories or files
        """
        energy_data = self._load_data_from_dir(settings.DAILYBLOCKS_DIR)
        weather_data = self._load_data_from_file(settings.WEATHER_DAILY_FILE)
        temperature_patterns = self._process_temperature_energy_patterns(
            energy_data, weather_data
        )

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(temperature_patterns)

        return temperature_patterns

    def get_temperature_hourly_patterns(self) -> pl.DataFrame:
        """
        Process hourly energy consumption data to analyze how it correlates with hourly temperature.

        This method serves as a pipeline that:
        1. Loads half-hourly energy data from the configured directory (HHBLOCKS_DIR)
        2. Loads hourly weather data from the configured file (WEATHER_HOURLY_FILE)
        3. Processes the data to extract hourly energy consumption patterns by temperature ranges
        4. Optionally displays debug information if enabled in settings

        Returns:
            Polars DataFrame containing hourly temperature-energy statistics with columns:
            - hour: Hour of day (0-23) (int)
            - temp_bin: Temperature range bin (str)
            - avg_temperature: Average temperature in that bin (float)
            - avg_humidity: Average humidity in that bin (float)
            - energy: Average energy consumption (float)
            - count: Number of data points in each bin (int)

        Raises:
            ValueError: If no valid CSV files are found in the configured directories or files
        """
        energy_data = self._load_data_from_dir(settings.HHBLOCKS_DIR)
        weather_data = self._load_data_from_file(settings.WEATHER_HOURLY_FILE)
        hourly_patterns = self._process_temperature_hourly_energy(energy_data, weather_data)

        if settings.DEBUG:
            with pl.Config(tbl_rows=-1, tbl_cols=-1):
                print(hourly_patterns)

        return hourly_patterns
