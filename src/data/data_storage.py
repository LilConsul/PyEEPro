from typing import List
import polars as pl
from data.cache_manager import CacheManager
from data.data_processor import DataProcessor


class DataStorage:
    """
    Singleton class that manages energy consumption pattern data with caching capabilities.

    This class serves as a centralized data access layer that:
    1. Retrieves energy consumption patterns (hourly, daily, weekly)
    2. Implements caching to avoid redundant data processing
    3. Provides filtering capabilities by year and columns

    The class uses DataProcessor for data transformation and CacheManager for
    persistent storage. The singleton pattern ensures that only one instance
    of DataStorage exists throughout the application lifecycle.

    Attributes:
        _instance (DataStorage): Singleton instance reference
        _processor (DataProcessor): Data processing component
        _cache_manager (CacheManager): Cache management component
        hourly_patterns (pl.DataFrame): Cached hourly patterns data
        daily_patterns (pl.DataFrame): Cached daily patterns data
        weekly_patterns (pl.DataFrame): Cached weekly patterns data
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Implement singleton pattern to ensure only one DataStorage instance exists.

        Returns:
            DataStorage: The singleton instance of DataStorage
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize DataStorage with processor and cache manager components.

        The initialization only happens once due to the singleton pattern.
        """
        self._processor = DataProcessor()
        self._cache_manager = CacheManager()

    def _get_patterns(
        self,
        pattern_attr: str,
        processor_method: callable,
        years: List[int] | None = None,
        cols: List[str] | None = None,
        cache_file: str | None = None,
    ) -> pl.DataFrame:
        """
        Generic wrapper for retrieving pattern data with filtering options.

        This method implements the caching strategy:
        1. Try to retrieve data from instance attribute
        2. If not available, try to load from disk cache
        3. If still not available, process the data and save to cache
        4. Apply any filtering by years or columns

        Args:
            pattern_attr: Name of the instance attribute to store the patterns
            processor_method: Method to call if data needs to be processed
            years: Optional list of years to filter by
            cols: Optional list of columns to select
            cache_file: Name of the cache file (defaults to pattern_attr + .csv)

        Returns:
            pl.DataFrame: DataFrame with the requested patterns, filtered if specified
        """
        if cache_file is None:
            cache_file = f"{pattern_attr}.csv"

        if not hasattr(self, pattern_attr):
            setattr(self, pattern_attr, self._cache_manager.load_cache(cache_file))

        data = getattr(self, pattern_attr)
        if data is None:
            data = processor_method()
            setattr(self, pattern_attr, data)
            self._cache_manager.save_cache(data, cache_file)

        result = data

        if years is not None and len(years) != 0:
            result = result.filter(pl.col("year").is_in(years))

        if cols is not None and len(cols) != 0:
            result = result.select(cols)

        return result

    def get_hourly_patterns(
        self, years: List[int] | None = None, cols: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Retrieve hourly patterns from cache or process if not available.

        This method provides access to hourly energy consumption patterns,
        with optional filtering by years and columns. The data is cached
        to avoid redundant processing.

        Args:
            years: Optional list of years to filter by (e.g., [2013, 2014])
            cols: Optional list of columns to select (e.g., ["year", "hour", "energy_mean"])

        Returns:
            pl.DataFrame: DataFrame with hourly patterns containing columns:
                - year: Calendar year (int)
                - hour: Hour of day (0-23) (int)
                - energy_median: Median energy consumption (float)
                - energy_mean: Mean energy consumption (float)
                - energy_max: Maximum energy consumption (float)
                - energy_count: Number of data points (int)
                - energy_std: Standard deviation (float)
                - energy_sum: Total energy consumption (float)
                - energy_min: Minimum energy consumption (float)
        """
        return self._get_patterns(
            "hourly_patterns",
            self._processor.get_hourly_patterns,
            years,
            cols,
        )

    def get_daily_patterns(
        self, years: List[int] | None = None, cols: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Retrieve daily patterns from cache or process if not available.

        This method provides access to daily energy consumption patterns by weekday,
        with optional filtering by years and columns. The data is cached
        to avoid redundant processing.

        Args:
            years: Optional list of years to filter by (e.g., [2013, 2014])
            cols: Optional list of columns to select (e.g., ["year", "weekday", "energy_mean"])

        Returns:
            pl.DataFrame: DataFrame with daily patterns containing columns:
                - year: Calendar year (int)
                - weekday: Day of week (1-7, Monday-Sunday) (int)
                - weekday_name: Name of day ("Monday" through "Sunday") (str)
                - energy_median: Average of daily median values (float)
                - energy_mean: Average of daily mean values (float)
                - energy_max: Maximum energy consumption (float)
                - energy_count: Sum of daily count values (int)
                - energy_std: Average of daily standard deviation (float)
                - energy_sum: Total energy consumption (float)
                - energy_min: Minimum energy consumption (float)
                - days_count: Number of days in each group (int)
        """
        return self._get_patterns(
            "daily_patterns",
            self._processor.get_daily_patterns,
            years,
            cols,
        )

    def get_weekly_patterns(
        self, years: List[int] | None = None, cols: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Retrieve weekly patterns from cache or process if not available.

        This method provides access to weekly energy consumption patterns,
        with optional filtering by years and columns. The data is cached
        to avoid redundant processing.

        Args:
            years: Optional list of years to filter by (e.g., [2013, 2014])
            cols: Optional list of columns to select (e.g., ["year", "week", "energy_mean"])

        Returns:
            pl.DataFrame: DataFrame with weekly patterns containing columns:
                - year: Calendar year (int)
                - week: Week number (1-53) (int)
                - energy_median: Average of daily median values (float)
                - energy_mean: Average of daily mean values (float)
                - energy_max: Maximum energy consumption (float)
                - energy_count: Sum of daily count values (int)
                - energy_std: Average of daily standard deviation (float)
                - energy_sum: Total energy consumption (float)
                - energy_min: Minimum energy consumption (float)
                - weeks_count: Number of days in each weekly group (int)
        """
        return self._get_patterns(
            "weekly_patterns",
            self._processor.get_weekly_patterns,
            years,
            cols,
        )

    def remove_cache(self, filename: str) -> None:
        """
        Remove a specific cache file and clear its in-memory representation.

        This method deletes both the persisted cache file and the corresponding
        in-memory attribute to ensure consistency.

        Args:
            filename: Name of the cache file to remove (e.g., "hourly_patterns.csv")

        Notes:
            - Attempts to clear the corresponding instance attribute based on filename
            - Delegates actual file removal to the cache manager
        """
        try:
            setattr(self, filename.split(".")[0], None)
        except AttributeError:
            print(f"Attribute '{filename.split('.')[0]}' does not exist.")
        self._cache_manager.remove_cache(filename)

    def remove_all_caches(self) -> None:
        """
        Remove all cache files and clear all in-memory pattern data.

        This method provides a complete reset of the cache state, both
        on disk and in memory. It preserves the processor and cache manager
        instances.

        Notes:
            - All instance attributes except _processor and _cache_manager are set to None
            - Delegates file removal to the cache manager
        """
        try:
            for attr in list(self.__dict__):
                if attr not in ["_processor", "_cache_manager"]:
                    setattr(self, attr, None)
        except Exception as e:
            print(f"Error removing attributes: {str(e)}")
        self._cache_manager.remove_all_caches()

    def get_cached_files(self) -> List[str]:
        """
        Get a list of all cached files.

        This method provides visibility into what pattern data is currently
        cached on disk.

        Returns:
            List[str]: List of cache filenames (e.g., ["hourly_patterns.csv",
            "daily_patterns.csv"])
        """
        return self._cache_manager.get_cached_files()


storage = DataStorage()


if __name__ == "__main__":
    storage = DataStorage()
    print(storage.get_weekly_patterns())
    print(storage.get_weekly_patterns())
    print(storage.get_daily_patterns())
    print(storage.get_daily_patterns())
    print(storage.get_hourly_patterns())
    print(storage.get_hourly_patterns())
