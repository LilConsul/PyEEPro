import sys
import time
from pathlib import Path
from .extract_dataset import extract_dataset


def check_dataset_exists(progress_callback=None) -> tuple[bool, str]:
    """Check if dataset exists in the expected location
    
    Args:
        progress_callback: Optional callback function to report extraction progress
    
    Returns:
        Tuple of (success, message)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    folder_path = project_root / "data" / "smart-meters-in-london"
    zip_path = project_root / "data" / "smart-meters-in-london.zip"

    if folder_path.exists() and folder_path.is_dir():
        if progress_callback:
            progress_callback(1.0, "Dataset folder already exists")
            time.sleep(0.5)  # Give UI time to update
        return True, "Dataset folder found"

    if progress_callback:
        progress_callback(0.0, "Checking for dataset ZIP file...")
        time.sleep(0.5)  # Give UI time to update

    if zip_path.exists() and zip_path.is_file():
        try:
            if progress_callback:
                progress_callback(0.1, f"Found ZIP file: {zip_path.name}")
                time.sleep(0.5)  # Give UI time to update
            
            # Use the progress callback for extraction
            extract_dataset(zip_path, silent=False, progress_callback=progress_callback)
            
            # Double check extraction was successful
            if folder_path.exists() and folder_path.is_dir():
                return True, "Dataset extracted successfully"
            else:
                return False, "Extraction completed but dataset folder not found"
                
        except Exception as e:
            error_msg = f"Error extracting dataset: {e}"
            print(error_msg, file=sys.stderr)
            if progress_callback:
                progress_callback(0.0, f"Error: {error_msg}")
            return False, error_msg
    else:
        if progress_callback:
            progress_callback(0.0, "Dataset ZIP file not found")
        return False, "Neither dataset folder nor zip file found"
