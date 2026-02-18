# Clustering_Business Notebook Updates

## Overview
This document contains the complete code updates needed for the Clustering_Business notebook to properly load cached outputs from Representation_Extracting and perform GPU-accelerated clustering.

## Cell-by-Cell Updates

### Cell 5: Replace with "Load Cached Latents"
**Action**: Replace the entire cell content

```python
# -------------------------
# Load cached latent representations from Representation_Extracting
# -------------------------
from pathlib import Path
import numpy as np
import json

# Configuration - MUST match Representation_Extracting notebook
cache_dir = Path("/Workspace/Users/gpuchalski@kumc.edu/projects/PFTSleep/cache_dir")
encoder_name = "PFTSleep"
num_files = 6
frequency = 125
win_length = 750
hop_length = 750
max_seq_len_sec = 8 * 3600

# Build cache tag (same function as in Representation_Extracting)
def cache_tag(encoder_name, num_files, frequency, win_length, hop_length, max_seq_len_sec):
    return f"{encoder_name}__files{num_files}__freq{frequency}__win{win_length}__hop{hop_length}__max{max_seq_len_sec}"

tag = cache_tag(encoder_name, num_files, frequency, win_length, hop_length, max_seq_len_sec)

# Load all cached arrays
Z = np.load(cache_dir / f"Z__{tag}.npy")
night_id = np.load(cache_dir / f"night_id__{tag}.npy")
time_idx = np.load(cache_dir / f"time_idx__{tag}.npy")
zarr_file_idx = np.load(cache_dir / f"zarr_file_idx__{tag}.npy")
windows_start_idx = np.load(cache_dir / f"windows_start_idx__{tag}.npy")
zarr_files_list = np.load(cache_dir / f"zarr_files_list__{tag}.npy", allow_pickle=True).tolist()

# Load zarr ID mapping
with open(cache_dir / "zarr_id_map.json", "r") as f:
    zarr_id_map = json.load(f)

print(f"✅ Loaded latent representations from Representation_Extracting:")
print(f"  Z shape: {Z.shape}")
print(f"  Night IDs: {len(np.unique(night_id))} unique nights")
print(f"  Zarr files: {len(zarr_files_list)} files")
print(f"  Zarr file indices: {len(np.unique(zarr_file_idx))} unique")
print(f"  Time indices range: {time_idx.min():.1f}s to {time_idx.max():.1f}s")
print(f"\n📁 Zarr files loaded:")
for uid, idx in sorted(zarr_id_map.items(), key=lambda x: x[1]):
    print(f"  [{idx}] {uid}")
```

### Cell 6: DELETE THIS CELL
**Action**: Delete the CuPy conversion cell - not needed since we load numpy arrays directly

### Cell 7: Update normalization
**Action**: Replace with simplified version (no CuPy conversion needed)

```python
# ------------------------------------------------------------
# 1) Normalize RAW 512-D (angular geometry)
# ------------------------------------------------------------
from sklearn.preprocessing import normalize

X = normalize(Z.astype(np.float32), norm="l2")
N, D = X.shape

print(f"✅ Normalized latent space ready for clustering:")
print(f"  Shape: {X.shape}")
print(f"  Data type: {X.dtype}")
```

### Cell 13: Replace with "Save Cluster Assignments"
**Action**: Replace entire cell content

```python
# -------------------------
# Save cluster assignments with zarr file mapping
# -------------------------
import pandas as pd

# Create comprehensive mapping dataframe
cluster_mapping = pd.DataFrame({
    'window_idx': np.arange(len(cluster_id)),
    'cluster_id': cluster_id,
    'zarr_file_idx': zarr_file_idx,
    'night_id': night_id,
    'time_idx_sec': time_idx,
    'windows_start_idx': windows_start_idx
})

# Add zarr file metadata
idx_to_uid = {v: k for k, v in zarr_id_map.items()}
cluster_mapping['zarr_uid'] = cluster_mapping['zarr_file_idx'].map(idx_to_uid)
cluster_mapping['zarr_file_path'] = cluster_mapping['zarr_file_idx'].apply(
    lambda x: zarr_files_list[x] if x < len(zarr_files_list) else None
)

# Save to cache directory
output_path = cache_dir / f"cluster_assignments__{tag}.csv"
cluster_mapping.to_csv(output_path, index=False)

print(f"✅ Saved cluster assignments to: {output_path}")
print(f"\n📊 Cluster distribution:")
print(cluster_mapping['cluster_id'].value_counts().sort_index())
print(f"\n📁 Data points per zarr file:")
zarr_summary = cluster_mapping.groupby('zarr_uid').agg({
    'window_idx': 'count',
    'cluster_id': lambda x: x.nunique()
}).rename(columns={'window_idx': 'num_windows', 'cluster_id': 'num_clusters'})
print(zarr_summary)

# Display sample
print(f"\n🔍 Sample cluster assignments:")
display(cluster_mapping.head(20))
```

### Cell 14: DELETE THIS CELL
**Action**: Delete the multi-file function cell - no longer needed

## Summary of Changes

1. ✅ **Cell 5**: Load cached latents from Representation_Extracting
2. ❌ **Cell 6**: DELETE (CuPy conversion not needed)
3. ✅ **Cell 7**: Simplified normalization (direct from numpy)
4. ✅ **Cell 13**: Complete cluster assignment mapping and saving
5. ❌ **Cell 14**: DELETE (redundant multi-file function)

## Workflow After Updates

1. Run **Representation_Extracting** notebook first
   - Extracts latents and saves to cache_dir
   - Creates zarr_id_map.json
   - Saves all metadata arrays

2. Run **Clustering_Business** notebook
   - Loads cached latents (Cell 5)
   - Normalizes data (Cell 7)
   - Performs LSH hashing (Cell 8)
   - Runs LSH-means initialization (Cell 9)
   - Sweeps K values with GPU KMeans (Cell 10)
   - Final clustering with best K (Cell 11)
   - UMAP visualization (Cell 12)
   - Saves cluster assignments with zarr mapping (Cell 13)

## Output Files

After running Clustering_Business, you'll have:
- `cluster_assignments__{tag}.csv` - Complete mapping of windows to clusters and source zarr files

Columns in CSV:
- `window_idx`: Sequential window index
- `cluster_id`: Assigned cluster ID
- `zarr_file_idx`: Numeric zarr file ID
- `night_id`: Night/batch identifier
- `time_idx_sec`: Time index in seconds
- `windows_start_idx`: Start index in original zarr
- `zarr_uid`: Original zarr file UID
- `zarr_file_path`: Full path to source zarr file
