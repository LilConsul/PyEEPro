import polars as pl
from src.app.config import setting


class DataProcessor:
    @staticmethod
    def load_data_from_dir(directory):
        """
        Load all CSV files from a directory and concatenate them into a single Polars DataFrame.

        Args:
            directory: Path to the directory containing CSV files.

        Returns:
            Polars DataFrame containing all records from the CSV files.
        """
        processed_data = []
        for file in directory.glob("*.csv"):
            try:
                data = pl.read_csv(file)
                processed_data.append(data)
                print(f"Loaded {file.name} with {len(data)} records")
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
        return pl.concat(processed_data)

    @staticmethod
    def convert_half_hourly_to_hourly(df):
        """
        Convert half-hourly electricity consumption data to hourly data by summing adjacent columns.

        Args:
            df: Polars DataFrame with columns LCLid, day, hh_0, hh_1, ..., hh_47

        Returns:
            Polars DataFrame with columns LCLid, day, h_0, h_1, ..., h_23
        """
        # Get the base columns to keep
        base_cols = ["LCLid", "day"]

        hourly_expressions = [
            (pl.col(f"hh_{hour * 2}") + pl.col(f"hh_{hour * 2 + 1}")).alias(f"h_{hour}")
            for hour in range(24)
        ]
        result = df.select([*base_cols, *hourly_expressions])

        return result

    @staticmethod
    def analyze_hourly_patterns(df):
        """
        Analyze hourly energy consumption patterns from hourly data.

        Args:
            df: Polars DataFrame with columns LCLid, day, h_0, h_1, ..., h_23

        Returns:
            Polars DataFrame with hourly average consumption across all houses/days
        """
        # Calculate mean consumption for each hour across all days and LCLids
        hourly_cols = [f"h_{hour}" for hour in range(24)]

        # Melt the dataframe to transform from wide to long format for analysis
        long_df = df.unpivot(
            index=["LCLid", "day"],
            on=hourly_cols,
            variable_name="hour",
            value_name="consumption",
        )

        # Extract the hour number from the column name
        long_df = long_df.with_columns(
            pl.col("hour").str.replace("h_", "").cast(pl.Int32).alias("hour_num")
        )

        # Aggregate to get average consumption by hour
        hourly_pattern = (
            long_df.group_by("hour_num")
            .agg(pl.mean("consumption").alias("avg_consumption"))
            .sort("hour_num")
        )

        return hourly_pattern

    def process(self):
        data = self.load_data_from_dir(
            setting.DATA_DIR / "hhblock_dataset" / "hhblock_dataset"
        )
        hourly_data = self.convert_half_hourly_to_hourly(data)
        hourly_patterns = self.analyze_hourly_patterns(hourly_data)
        print(hourly_patterns)
        return hourly_patterns


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process()
