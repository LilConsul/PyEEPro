from typing import List
import polars as pl
from data.cache_manager import CacheManager
from data.data_processor import DataProcessor


class DataStorage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._processor = DataProcessor()
        self._cache_manager = CacheManager()
        self.hourly_patterns = self._cache_manager.load_cache("hourly_patterns.csv")
        self.dayly_patterns = self._cache_manager.load_cache("daily_patterns.csv")

    def _get_patterns(
        self,
        pattern_attr: str,
        cache_file: str,
        processor_method: callable,
        years: List[int] | None = None,
        cols: List[str] | None = None,
    ) -> pl.DataFrame:
        """
        Generic wrapper for retrieving pattern data with filtering options.

        Args:
            pattern_attr: Name of the attribute to store the patterns.
            cache_file: Name of the cache file.
            processor_method: Method to call if data needs to be processed.
            years: Optional list of years to filter by.
            cols: Optional list of columns to select.

        Returns:
            DataFrame with the requested patterns.
        """
        # Check if data is already loaded
        data = getattr(self, pattern_attr)
        if data is None:
            # Process and cache data if not available
            data = processor_method()
            setattr(self, pattern_attr, data)
            self._cache_manager.save_cache(data, cache_file)

        result = data

        # Apply filters
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

        Args:
            years: Optional list of years to filter by.
            cols: Optional list of columns to select.

        Returns:
            DataFrame with hourly patterns.
        """
        return self._get_patterns(
            "hourly_patterns",
            "hourly_patterns.csv",
            self._processor.get_hourly_patterns,
            years,
            cols,
        )

    def get_daily_patterns(
        self, years: List[int] | None = None, cols: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Retrieve daily patterns from cache or process if not available.

        Args:
            years: Optional list of years to filter by.
            cols: Optional list of columns to select.

        Returns:
            DataFrame with daily patterns.
        """
        return self._get_patterns(
            "dayly_patterns",  # Using existing attribute name for consistency
            "daily_patterns.csv",
            self._processor.get_dyily_patterns,
            years,
            cols,
        )

    def remove_cache(self, filename: str) -> None:
        """Remove a specific cache file."""
        try:
            setattr(self, filename.split(".")[0], None)
        except AttributeError:
            print(f"Attribute '{filename.split('.')[0]}' does not exist.")
        self._cache_manager.remove_cache(filename)

    def remove_all_caches(self) -> None:
        """Remove all cache files."""
        try:
            for attr in list(self.__dict__):
                if attr not in ["_processor", "_cache_manager"]:
                    setattr(self, attr, None)
        except Exception as e:
            print(f"Error removing attributes: {str(e)}")
        self._cache_manager.remove_all_caches()

    def get_cached_files(self) -> List[str]:
        """Get a list of all cached files."""
        return self._cache_manager.get_cached_files()


storage = DataStorage()


if __name__ == "__main__":
    storage = DataStorage()
    print(storage.get_hourly_patterns())
    print(storage.get_daily_patterns())
    storage.remove_cache('hourly_patterns.csv')
