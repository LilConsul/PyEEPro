import sys
import os
from pathlib import Path
from .extract_dataset import extract_dataset


def check_dataset_exists(progress_callback=None) -> tuple[bool, str]:
    """Check if dataset exists in the expected location
    
    Args:
        progress_callback: Optional callback function to report extraction progress
    
    Returns:
        Tuple of (success, message)
    """
    # Get project root (two levels up from scripts folder)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # Going up one more level to reach project root

    # Use absolute paths to avoid any path resolution issues
    folder_path = project_root / "data" / "smart-meters-in-london"
    zip_path = project_root / "data" / "smart-meters-in-london.zip"

    # Report start of check with absolute paths
    if progress_callback:
        progress_callback(0.0, f"Checking for dataset at {folder_path}...")

    # Check if dataset folder exists
    if folder_path.exists() and folder_path.is_dir():
        # Dataset already extracted
        if progress_callback:
            progress_callback(1.0, "Dataset already exists in folder")
        return True, "Dataset folder found"

    # Report looking for ZIP with absolute path
    if progress_callback:
        progress_callback(0.1, f"Looking for ZIP file at {zip_path}...")

    # Check if ZIP file exists
    if zip_path.exists() and zip_path.is_file():
        try:
            # Verify ZIP file is readable
            if os.access(zip_path, os.R_OK):
                if progress_callback:
                    size_mb = zip_path.stat().st_size / (1024 * 1024)
                    progress_callback(0.2, f"Found ZIP file: {zip_path.name} ({size_mb:.1f} MB)")
            else:
                if progress_callback:
                    progress_callback(0.0, f"ZIP file found but cannot be read - check permissions")
                return False, "ZIP file found but cannot be read due to permissions"
            
            # Extract the dataset with progress reporting
            extract_dataset(str(zip_path.resolve()), silent=False, progress_callback=progress_callback)
            
            # Verify extraction was successful
            if folder_path.exists() and folder_path.is_dir():
                if progress_callback:
                    progress_callback(1.0, "Dataset extracted successfully")
                return True, "Dataset extracted successfully"
            else:
                if progress_callback:
                    progress_callback(0.0, "Extraction completed but dataset folder not found")
                return False, "Extraction completed but dataset folder not found"
                
        except Exception as e:
            error_msg = f"Error extracting dataset: {e}"
            print(error_msg, file=sys.stderr)
            if progress_callback:
                progress_callback(0.0, error_msg)
            return False, error_msg
    else:
        # No ZIP file found
        if progress_callback:
            progress_callback(0.0, f"Dataset ZIP file not found at {zip_path}")
            # Check if data directory exists
            data_dir = project_root / "data"
            if data_dir.exists():
                files = list(data_dir.glob("*.zip"))
                if files:
                    progress_callback(0.0, f"Found other ZIP files: {', '.join(f.name for f in files)}")
        return False, f"Neither dataset folder nor zip file found. Expected at {zip_path}"
