from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    ALL_DATA_DIR = BASE_DIR / "data"
    DATA_DIR = ALL_DATA_DIR / "smart-meters-in-london"
    HHBLOCKS_DIR = DATA_DIR / "hhblock_dataset" / "hhblock_dataset"
    DYLYBLOCKS_DIR = DATA_DIR / "daily_dataset" / "daily_dataset"
    CACHE_DIR = ALL_DATA_DIR / "cache"
    DEBUG = False


settings = Settings()
