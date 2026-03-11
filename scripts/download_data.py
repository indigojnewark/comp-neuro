"""Download flanker dataset."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import download_flanker_data, get_dataset_info, load_flanker_data

def main():
    print("Downloading Hedge et al. (2018) flanker dataset...\n")
    download_flanker_data(force_download=False)
    
    print("\nLoading and validating data...")
    df = load_flanker_data(download_if_missing=False)
    info = get_dataset_info(df)
    
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nData download complete! Ready for analysis.")

if __name__ == "__main__":
    main()
