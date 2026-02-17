import subprocess
import sys

# Install compatible zarr version
subprocess.check_call([sys.executable, "-m", "pip", "install", "zarr<3", "-q"])

from pftsleep.slumber import edf_signals_to_zarr

from pathlib import Path
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

write_data_dir = "/Workspace/Users/gpuchalski@kumc.edu/projects/PFTSleep/zarrs"
edf_files = glob.glob("/Volumes/kumc_sleep/sleep_studies/shhs_data/pftsleep/*.edf")
current_zarr_files = glob.glob(str(Path(write_data_dir)/"*.zarr"))

df = pd.DataFrame(edf_files, columns=['file_path'])
df['file_name'] = df['file_path'].apply(lambda x: Path(x).stem)
df['zarr_exists'] = df['file_name'].isin(map(lambda x: Path(x).stem, current_zarr_files))
# these are the files to process (havent been processed yet)
edf_files = df.loc[df['zarr_exists'] == False, 'file_path'].unique().tolist()

## NOTE THAT WHILE I DID NOT MAP HYPNOGRAMS, I DID PERFORM A MAPPING AFTER THE FACT, using the following code:
# def convert_hypnogram(hyp):
#     hyp[hyp>5] = -100
#     hyp[hyp>=4] = hyp[hyp>=4] - 1
#     return hyp
#for z in tqdm(zarr_files_shhs):
    # rt = zarr.open(z, mode='r')
    # hyp = rt['hypnogram'][:]
    # hyp = convert_hypnogram(hyp)
    # a = da.from_array(hyp, chunks='auto')
    # a.to_zarr(url=rt.store, component='hypnogram', compute=True, overwrite=True)

def process_file(file, frequency=None, write_data_dir=write_data_dir):
    try:
        _ = edf_signals_to_zarr(file, frequency=frequency, write_data_dir=write_data_dir)
        return True, file, None
    except Exception as e:
        return False, file, str(e)


if __name__ == '__main__':
    total_files = len(edf_files)
    print(f'Found {total_files} files to process')
    print(f'Beginning processing with 4 threads')
    
    completed = 0
    failed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, file): file for file in edf_files}
        
        for future in as_completed(futures):
            file = futures[future]
            try:
                success, processed_file, error = future.result()
                if success:
                    completed += 1
                    print(f"Progress: {completed}/{total_files} completed ({failed} failed) | File: {Path(processed_file).name}", flush=True)
                else:
                    failed += 1
                    print(f"Error parsing file: {Path(processed_file).name}. Error: {error}", flush=True)
            except Exception as e:
                failed += 1
                print(f"Unexpected error with file: {Path(file).name}. Error: {e}", flush=True)
    
    print('Job Completed')
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    print(f"Successfully processed {completed}/{total_files} files ({failed} failed)")