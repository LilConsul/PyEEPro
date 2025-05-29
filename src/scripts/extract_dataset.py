import sys
import zipfile
import time
from pathlib import Path
from typing import Callable, Optional


def extract_dataset(
    dataset_zip_path: Path = Path("./data/smart-meters-in-london.zip"),
    silent=False,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> None:
    try:
        if isinstance(dataset_zip_path, str):
            dataset_zip_path = Path(dataset_zip_path)

        output_dir = dataset_zip_path.parent / dataset_zip_path.stem

        if not dataset_zip_path.is_file():
            if not silent:
                print(
                    f"Dataset zip file {dataset_zip_path} does not exist.",
                    file=sys.stderr,
                )
            return

        output_dir.mkdir(exist_ok=True, parents=True)

        if not silent:
            print(f"Extracting dataset from {dataset_zip_path} to {output_dir}...")

        # Report initial progress
        if progress_callback:
            progress_callback(0.0, f"Starting extraction of {dataset_zip_path.name}...")
            # Small delay to allow UI to update
            time.sleep(0.1)
        
        # Extract the dataset with progress reporting
        try:
            with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                # Get total files and size for progress calculation
                file_list = zip_ref.infolist()
                total_files = len(file_list)
                total_size = sum(file.file_size for file in file_list)
                extracted_files = 0
                extracted_size = 0
                
                # Report number of files to extract
                if progress_callback:
                    progress_callback(0.0, f"Found {total_files} files to extract (total size: {total_size/1024/1024:.2f} MB)")
                    # Small delay to allow UI to update
                    time.sleep(0.1)
                
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, output_dir)
                    extracted_files += 1
                    extracted_size += file.file_size
                    
                    # Report progress if callback is provided - use either file count or size based progress
                    if progress_callback and (i % 5 == 0 or i == total_files - 1):  # Update every 5 files to avoid too many UI updates
                        # Use file size for more accurate progress percentage
                        progress = min(extracted_size / total_size, 0.99) if total_size > 0 else extracted_files / total_files
                        file_name = file.filename.split('/')[-1] if '/' in file.filename else file.filename
                        status_text = f"Extracting: {file_name} ({extracted_files}/{total_files}, {extracted_size/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)"
                        progress_callback(progress, status_text)
                    
                    # Basic progress reporting to console if not silent
                    if not silent and extracted_files % 20 == 0:  # Report every 20 files
                        print(f"Progress: {extracted_files}/{total_files} files extracted ({extracted_size/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)")
            
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            error_msg = f"ZIP file error: {e}"
            if progress_callback:
                progress_callback(0.0, f"Error: {error_msg}")
            raise ValueError(error_msg)

        if not silent:
            print("Extraction complete.")
            print(f"Dataset extracted to {output_dir}")
            
        # Final progress update
        if progress_callback:
            progress_callback(1.0, f"Extraction complete. Files saved to {output_dir.name}")

    except Exception as e:
        if not silent:
            print(f"An error occurred while extracting the dataset: {e}", file=sys.stderr)
        else:
            sys.stderr.write(f"Error: {e}\n")
            
        # Report error in progress
        if progress_callback:
            progress_callback(0.0, f"Error during extraction: {str(e)}")
            
        raise  # Re-raise the exception to be handled by the caller


if __name__ == "__main__":
    zip_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("./data/smart_meters_in_london.zip")
    )
    extract_dataset(zip_path)
