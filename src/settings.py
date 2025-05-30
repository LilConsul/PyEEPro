from pathlib import Path
class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "smart-meters-in-london"
    HHBLOCKS_DIR = DATA_DIR / "hhblock_dataset" / "hhblock_dataset"
    DEBUG = True


settings = Settings()
