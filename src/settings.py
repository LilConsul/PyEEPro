from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    ALL_DATA_DIR = BASE_DIR / "data"
    DATA_DIR = ALL_DATA_DIR / "smart-meters-in-london"
    HHBLOCKS_DIR = DATA_DIR / "hhblock_dataset" / "hhblock_dataset"
    DAILYBLOCKS_DIR = DATA_DIR / "daily_dataset" / "daily_dataset"
    HOLIDAYS_FILE = DATA_DIR / "uk_bank_holidays.csv"
    INFORMATION_HOUSEHOLD_FILE = DATA_DIR / "informations_households.csv"
    WEATHER_DAILY_FILE = DATA_DIR / "weather_daily_darksky.csv"
    WEATHER_HOURLY_FILE = DATA_DIR / "weather_hourly_darksky.csv"
    CACHE_DIR = ALL_DATA_DIR / "cache"
    MODEL_DIR = CACHE_DIR / "models"
    DEBUG = True


settings = Settings()
