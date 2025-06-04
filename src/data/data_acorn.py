import polars as pl
from settings import settings
from data.cache_manager import CacheManager


class AcornData:
    """
    day_of_week (0–6),
    is_weekend (bool),
    avg_temperature (float),
    avg_humidity(float),
    season (int),
    hh_consumption (list[float])
    """

    def __init__(self, acorn_group, selected_years: (int, int) = (2011, 2021)):
        """
        Initializes the AcornData class with a specific acorn group.
        """
        self.cache_manager = CacheManager(cache_dir=settings.CACHE_DIR / "acorn_data")
        self.acorn_group = acorn_group
        self.files = (
            pl.read_csv(settings.INFORMATION_HOUSEHOLD_FILE)
            .filter(pl.col("Acorn_grouped") == self.acorn_group)
            .select(pl.col("file"))
            .unique()
        )
        if self.files.is_empty():
            raise ValueError(f"No data found for Acorn group: {self.acorn_group}")
        self.selected_years = selected_years
        self._weather_data = pl.read_csv(settings.WEATHER_DAILY_FILE)
        self._holidays_data = pl.read_csv(settings.HOLIDAYS_FILE)

    def get_data(self) -> pl.DataFrame:
        """
        Retrieves the processed data for the specified acorn group.
        :return: Polars DataFrame with processed data.
        """
        cache_file = f"acorn_data_{self.acorn_group}_({self.selected_years[0]}-{self.selected_years[1]}).csv"
        cached_data = self.cache_manager.load_cache(cache_file)
        if cached_data is not None:
            return cached_data

        df = self._process_data()
        self.cache_manager.save_cache(df, cache_file)
        return df

    def _process_data(self) -> pl.DataFrame:
        result_frames = []

        weather_daily = (
            self._weather_data.with_columns(
                pl.col("time")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                .dt.date()
                .alias("date")
            )
            .group_by("date")
            .agg(
                [
                    pl.mean("temperatureMax").alias("avg_temperature"),
                    pl.mean("humidity").alias("avg_humidity"),
                ]
            )
        )

        holidays_df = (
            self._holidays_data.with_columns(
                pl.col("Bank holidays").str.to_date().alias("holiday_date")
            )
            .select("holiday_date")
            .with_columns(pl.lit(True).alias("is_holiday"))
        )

        for file in self.files["file"].to_list():
            file_path = settings.HHBLOCKS_DIR / (file + ".csv")
            if not file_path.exists():
                raise FileNotFoundError(
                    f"File {file} does not exist in {settings.HHBLOCKS_DIR}"
                )

            df = pl.read_csv(file_path)
            df = df.with_columns(pl.col("day").str.to_date())
            df = df.filter(
                pl.col("day")
                .dt.year()
                .is_between(self.selected_years[0], self.selected_years[1])
            )

            # Add day of week
            df = df.with_columns(pl.col("day").dt.weekday().alias("day_of_week"))

            df = df.join(
                holidays_df, left_on="day", right_on="holiday_date", how="left"
            ).with_columns(
                pl.col("is_holiday").fill_null(False)
            )

            df = df.with_columns(
                ((pl.col("day").dt.weekday() >= 5) | pl.col("is_holiday")).alias(
                    "is_weekend"
                )
            )

            df = df.join(weather_daily, left_on="day", right_on="date", how="left")

            df = df.with_columns(((pl.col("day").dt.month() % 12) // 3).alias("season"))

            consumption_cols = [col for col in df.columns if col.startswith("hh")]
            df = df.with_columns(
                pl.concat_list(consumption_cols).alias("hh_consumption")
            )

            df = df.select(
                [
                    "day_of_week",
                    "is_weekend",
                    "avg_temperature",
                    "avg_humidity",
                    "season",
                    "hh_consumption",
                ]
            )
            result_frames.append(df)

        if result_frames:
            return pl.concat(result_frames)
        else:
            return pl.DataFrame(
                schema={
                    "day_of_week": pl.Int32,
                    "is_weekend": pl.Boolean,
                    "avg_temperature": pl.Float64,
                    "avg_humidity": pl.Float64,
                    "season": pl.Int32,
                    "hh_consumption": pl.List(pl.Float64),
                }
            )


if __name__ == "__main__":
    acorn_data = AcornData(acorn_group="Comfortable", selected_years=(2011, 2012))
    data = acorn_data.get_data()
    print(data)

