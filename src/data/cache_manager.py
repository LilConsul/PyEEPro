from typing import List
import polars as pl
from settings import settings


class CacheManager:
    """Manages caching operations for data frames"""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)

    def load_cache(self, filename: str) -> pl.DataFrame | None:
        """Load data from cache if it exists."""
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                return pl.read_csv(cache_path)
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error loading cache {filename}: {str(e)}")
        return None

    def save_cache(self, data: pl.DataFrame, filename: str) -> None:
        """Save processed data to cache."""
        cache_path = self.cache_dir / filename
        try:
            data.write_csv(cache_path)
            if settings.DEBUG:
                print(f"Cached data saved to {filename}")
        except Exception as e:
            if settings.DEBUG:
                print(f"Error saving cache {filename}: {str(e)}")

    def remove_cache(self, filename: str) -> None:
        """Remove a specific cache file."""
        cache_path = self.cache_dir / filename
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

    def remove_all_caches(self) -> None:
        """Remove all cache files."""
        if self.cache_dir.exists():
            try:
                for file in self.cache_dir.glob("*.csv"):
                    file.unlink()
                if settings.DEBUG:
                    print("All caches files removed")
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error removing caches: {str(e)}")
        else:
            if settings.DEBUG:
                print("Cache directory does not exist")

    def get_cached_files(self) -> List[str]:
        """Get a list of all cached files."""
        if self.cache_dir.exists():
            return [file.name for file in self.cache_dir.glob("*.csv")]
        return []