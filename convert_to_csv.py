import pandas as pd
import os
from glob import glob
import traceback

def convert_pkl_to_csv(data_dir):
    """Convert all pickle files in a directory to CSV format"""
    print(f"Starting conversion process in directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        return
    
    # Get all pickle files
    pickle_files = glob(os.path.join(data_dir, '*.pkl'))
    
    if not pickle_files:
        print(f"No pickle files found in {data_dir}")
        return
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Create a directory for CSV files if it doesn't exist
    csv_dir = os.path.join(data_dir, 'csv_files')
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created CSV output directory: {csv_dir}")
    
    # Convert each file
    success_count = 0
    error_count = 0
    
    for pkl_file in pickle_files:
        try:
            print(f"\nConverting {pkl_file}...")
            
            # Load the pickle file
            print("Loading pickle file...")
            df = pd.read_pickle(pkl_file)
            print(f"Successfully loaded data with shape: {df.shape}")
            
            # Create CSV filename
            base_name = os.path.basename(pkl_file)
            csv_file = os.path.join(csv_dir, base_name.replace('.pkl', '.csv'))
            
            # Save as CSV
            print(f"Saving to {csv_file}...")
            df.to_csv(csv_file, index=False)
            print(f"Successfully saved CSV file")
            success_count += 1
            
        except Exception as e:
            print(f"Error converting {pkl_file}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            error_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {success_count} files")
    print(f"Failed conversions: {error_count} files")

if __name__ == "__main__":
    data_dir = 'data'  # Directory containing pickle files
    convert_pkl_to_csv(data_dir) 