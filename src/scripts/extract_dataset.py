import sys
import os
import zipfile
import time
from pathlib import Path
from typing import Callable, Optional


def extract_dataset(
    dataset_zip_path: Path = Path("./data/smart-meters-in-london.zip"),
    silent=False,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> None:
    """Extract the dataset from a zip file with progress reporting.
    
    Args:
        dataset_zip_path: Path to the dataset zip file
        silent: If True, suppress output to console
        progress_callback: Function to report progress (takes progress percentage and status message)
    """
    try:
        # Convert to Path object and resolve to absolute path
        if isinstance(dataset_zip_path, str):
            dataset_zip_path = Path(dataset_zip_path).resolve()
        else:
            dataset_zip_path = dataset_zip_path.resolve()

        output_dir = dataset_zip_path.parent / dataset_zip_path.stem

        # Extra validation and checks
        if not dataset_zip_path.exists():
            error_msg = f"Dataset zip file does not exist at {dataset_zip_path}"
            if not silent:
                print(error_msg, file=sys.stderr)
            if progress_callback:
                progress_callback(0.0, error_msg)
            return
            
        if not dataset_zip_path.is_file():
            error_msg = f"Path exists but is not a file: {dataset_zip_path}"
            if not silent:
                print(error_msg, file=sys.stderr)
            if progress_callback:
                progress_callback(0.0, error_msg)
            return

        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)

        if not silent:
            print(f"Extracting dataset from {dataset_zip_path} to {output_dir}...")

        # Initial progress update
        if progress_callback:
            progress_callback(0.25, f"Starting extraction from {dataset_zip_path.name}...")
        
        # Verify file is a valid ZIP
        if not zipfile.is_zipfile(dataset_zip_path):
            error_msg = f"File is not a valid ZIP archive: {dataset_zip_path}"
            if progress_callback:
                progress_callback(0.0, error_msg)
            if not silent:
                print(error_msg, file=sys.stderr)
            return
        
        # Extract with progress reporting
        try:
            with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                # Get file list for progress tracking
                files = zip_ref.infolist()
                total_files = len(files)
                
                if total_files == 0:
                    if progress_callback:
                        progress_callback(0.0, f"ZIP file is empty: {dataset_zip_path.name}")
                    return
                
                total_size = sum(file.file_size for file in files)
                
                if progress_callback:
                    progress_callback(0.3, f"Found {total_files} files to extract ({total_size/1024/1024:.1f} MB total)")
                
                # Extract files with progress updates
                extracted_files = 0
                extracted_size = 0
                
                for i, file in enumerate(files):
                    try:
                        # Extract file
                        zip_ref.extract(file, output_dir)
                        extracted_files += 1
                        extracted_size += file.file_size
                        
                        # Calculate progress percentage (scale from 0.3 to 0.95)
                        progress = 0.3 + 0.65 * ((i + 1) / total_files)
                        
                        # Update progress every few files to avoid UI slowdown
                        if progress_callback and (i % 5 == 0 or i == total_files - 1):
                            filename = file.filename.split('/')[-1] if '/' in file.filename else file.filename
                            message = f"Extracting: {filename} ({extracted_files}/{total_files}, {extracted_size/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)"
                            progress_callback(progress, message)
                        
                        # Console logging
                        if not silent and i % 20 == 0:
                            print(f"Progress: {extracted_files}/{total_files} files extracted")
                    except Exception as e:
                        # Log the error but continue with other files
                        print(f"Error extracting file {file.filename}: {e}", file=sys.stderr)
                        if progress_callback:
                            progress_callback(progress, f"Warning: Error extracting {file.filename}: {e}")
                            time.sleep(0.5)  # Give time to read warning
        
        except zipfile.BadZipFile:
            error_msg = f"Error: File is not a valid ZIP file or is corrupted: {dataset_zip_path}"
            if progress_callback:
                progress_callback(0.0, error_msg)
            if not silent:
                print(error_msg, file=sys.stderr)
            raise ValueError(error_msg)

        # Final progress update
        if progress_callback:
            progress_callback(1.0, f"Extraction complete! {extracted_files} files extracted to {output_dir.name}")

        if not silent:
            print(f"Dataset extracted to {output_dir}")

    except PermissionError as pe:
        error_msg = f"Permission error: Unable to extract ZIP file. Check file permissions. {pe}"
        if not silent:
            print(error_msg, file=sys.stderr)
        if progress_callback:
            progress_callback(0.0, error_msg)
        raise
    except Exception as e:
        error_msg = f"Error extracting dataset: {e}"
        if not silent:
            print(error_msg, file=sys.stderr)
        if progress_callback:
            progress_callback(0.0, error_msg)
        raise


if __name__ == "__main__":
    # When running directly, look for the ZIP file at project root
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # Going up to the project root
    zip_path = project_root / "data" / "smart-meters-in-london.zip"
    extract_dataset(zip_path)
