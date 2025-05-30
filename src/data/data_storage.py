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
        self.processor = DataProcessor()
        self.cache_manager = CacheManager()
        self.hourly_patterns = self.cache_manager.load_cache("hourly_patterns.csv")

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
            self.cache_manager.save_cache(data, "hourly_patterns.csv")

        result = self.hourly_patterns

        if years is not None and len(years) != 0:
            result = result.filter(pl.col("year").is_in(years))

        if cols is not None and len(cols) != 0:
            result = result.select(cols)

        return result

    # Delegation methods for backward compatibility
    def remove_cache(self, filename: str) -> None:
        """Remove a specific cache file."""
        self.cache_manager.remove_cache(filename)

    def remove_all_caches(self) -> None:
        """Remove all cache files."""
        self.cache_manager.remove_all_caches()

    def get_cached_files(self) -> List[str]:
        """Get a list of all cached files."""
        return self.cache_manager.get_cached_files()


storage = DataStorage()


if __name__ == "__main__":
    storage = DataStorage()
    print(storage.get_hourly_patterns(years=[], cols=[]))