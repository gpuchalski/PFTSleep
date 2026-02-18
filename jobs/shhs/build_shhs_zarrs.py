import sys

# Install compatible zarr version
%pip install "zarr<3" -q

from pftsleep.slumber import edf_signals_to_zarr

from pathlib import Path
import glob
import time
import pandas as pd

write_data_dir = "/Workspace/Users/gpuchalski@kumc.edu/projects/PFTSleep/zarrs"
edf_files = glob.glob("/Volumes/kumc_sleep/sleep_studies/shhs_data/edfs_to_process/*.edf")
current_zarr_files = glob.glob(str(Path(write_data_dir)/"*.zarr"))

# DIAGNOSTIC: Show what was found
print(f"Total EDF files found in source directory: {len(edf_files)}")
print(f"Total zarr files already processed: {len(current_zarr_files)}")

df = pd.DataFrame(edf_files, columns=['file_path'])
df['file_name'] = df['file_path'].apply(lambda x: Path(x).stem)
df['zarr_exists'] = df['file_name'].isin(map(lambda x: Path(x).stem, current_zarr_files))

# DIAGNOSTIC: Show filtering results
print(f"Files already processed (zarr exists): {df['zarr_exists'].sum()}")
print(f"Files to process (zarr missing): {(~df['zarr_exists']).sum()}")

# these are the files to process (havent been processed yet)
edf_files = df.loc[df['zarr_exists'] == False, 'file_path'].unique().tolist()

def process_file(file, frequency=None, write_data_dir=write_data_dir):
    try:
        _ = edf_signals_to_zarr(file, frequency=frequency, write_data_dir=write_data_dir)
        return (file, True, None)
    except Exception as e:
        return (file, False, str(e))

if __name__ == '__main__':
    total_files = len(edf_files)
    print(f'\nFound {total_files} files to process')
    
    if total_files == 0:
        print("No files to process - all EDF files have already been converted to zarr format!")
        sys.exit(0)
    
    print('Beginning sequential processing...')

    start_time = time.time()
    
    results = []
    for i, file in enumerate(edf_files, 1):
        print(f"Processing file {i}/{total_files}: {Path(file).name}")
        result = process_file(file)
        results.append(result)
        
        # Show progress every 10 files
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_files - i) * avg_time
            print(f"  Progress: {i}/{total_files} ({100*i/total_files:.1f}%) - Est. remaining: {remaining/60:.1f} min")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=['file_path', 'success', 'error'])
    
    completed = results_df['success'].sum()
    failed = len(results_df) - completed
    
    print('\nJob Completed')
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    print(f"Successfully processed {completed}/{total_files} files ({failed} failed)")
    
    # Show any errors
    if failed > 0:
        print("\nFailed files:")
        for _, row in results_df[results_df['success'] == False].iterrows():
            print(f"  {Path(row['file_path']).name}: {row['error']}")