from typing import List
import polars as pl
from settings import settings
import json


class CacheManager:
    """Manages caching operations for data frames"""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)

    def save_cache(self, data: pl.DataFrame, filename: str) -> None:
        """Save processed data to cache."""
        cache_path = self.cache_dir / filename
        metadata_path = self.cache_dir / (filename + ".meta")

        # Get type information for all columns
        column_types = {}
        list_columns = []

        for col_name, dtype in data.schema.items():
            if isinstance(dtype, pl.List):
                list_columns.append(col_name)
                # Store the inner type of the list
                column_types[col_name] = {
                    "type": "list",
                    "inner_type": str(dtype.inner),
                }
            else:
                column_types[col_name] = {"type": str(dtype)}

        try:
            # Save metadata
            with open(metadata_path, "w") as f:
                json.dump(
                    {"column_types": column_types, "list_columns": list_columns}, f
                )

            # Handle list columns for CSV storage
            if list_columns:
                df_to_save = data.clone()
                for col in list_columns:
                    df_to_save = df_to_save.with_columns(
                        pl.col(col)
                        .map_elements(
                            lambda x: "|".join(map(str, x)), return_dtype=pl.Utf8
                        )
                        .alias(col)
                    )
                df_to_save.write_csv(cache_path)
            else:
                data.write_csv(cache_path)

            if settings.DEBUG:
                print(f"Cached data saved to {filename}")
        except Exception as e:
            if settings.DEBUG:
                print(f"Error saving cache {filename}: {str(e)}")

    def load_cache(self, filename: str) -> pl.DataFrame | None:
        """Load data from cache if it exists."""
        cache_path = self.cache_dir / filename
        metadata_path = self.cache_dir / (filename + ".meta")

        if cache_path.exists():
            try:
                df = pl.read_csv(cache_path)

                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    column_types = metadata.get("column_types", {})
                    list_columns = metadata.get("list_columns", [])

                    for col, type_info in column_types.items():
                        if col in df.columns:
                            if col in list_columns:
                                inner_type = type_info.get("inner_type", "Float64")
                                inner_dtype_name = inner_type.split("(")[0]
                                inner_dtype = getattr(pl, inner_dtype_name)

                                df = df.with_columns(
                                    pl.col(col)
                                    .str.split("|")
                                    .map_elements(
                                        lambda x: [
                                            float(val)
                                            for val in x
                                            if val and val.lower() != "none"
                                        ],
                                        return_dtype=pl.List(inner_dtype),
                                    )
                                    .alias(col)
                                )
                            else:
                                dtype_str = type_info.get("type")
                                if dtype_str:
                                    dtype_name = dtype_str.split("(")[0]
                                    if hasattr(pl, dtype_name):
                                        df = df.with_columns(
                                            pl.col(col).cast(getattr(pl, dtype_name))
                                        )

                return df
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error loading cache {filename}: {str(e)}")
        return None

    def remove_cache(self, filename: str) -> None:
        """Remove a specific cache file and its metadata."""
        cache_path = self.cache_dir / filename
        metadata_path = self.cache_dir / (filename + ".meta")

        if cache_path.exists():
            try:
                cache_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                if settings.DEBUG:
                    print(f"Cache {filename} removed")
            except Exception as e:
                if settings.DEBUG:
                    print(f"Error removing cache {filename}: {str(e)}")
        else:
            if settings.DEBUG:
                print(f"Cache {filename} does not exist")

    def remove_all_caches(self) -> None:
        """Remove all cache files and their metadata."""
        if self.cache_dir.exists():
            try:
                for file in self.cache_dir.glob("*.csv"):
                    file.unlink()

                for file in self.cache_dir.glob("*.meta"):
                    file.unlink()

                if settings.DEBUG:
                    print("All cache files removed")
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
