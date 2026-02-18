# REQUIRED UPDATE: Clustering_Business Notebook

## Status
⚠️ **ACTION REQUIRED**: Clustering_Business Cell 5 must be updated to match the new compressed format from Representation_Extracting.

## What Changed in Representation_Extracting (Cell 9)
✅ **COMPLETED** - The following changes fix the "File too large" error:
- File extension: `.npy` → `.npz`
- Save method: `np.save(Z_path, Z)` → `np.savez_compressed(Z_path, Z=Z)`
- Load method: `np.load(Z_path)` → `np.load(Z_path)['Z']`

## Required Change in Clustering_Business (Cell 5)

### Current Code (Line 24):
```python
Z = np.load(cache_dir / f"Z__{tag}.npy")
```

### Updated Code (Line 24):
```python
Z = np.load(cache_dir / f"Z__{tag}.npz")['Z']
```

## Complete Updated Cell 5 Code

Replace the entire Cell 5 content with:

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
num_files = 2
frequency = 125
win_length = 750
hop_length = 750
max_seq_len_sec = 8 * 3600

# Build cache tag (same function as in Representation_Extracting)
def cache_tag(encoder_name, num_files, frequency, win_length, hop_length, max_seq_len_sec):
    return f"{encoder_name}__files{num_files}__freq{frequency}__win{win_length}__hop{hop_length}__max{max_seq_len_sec}"

tag = cache_tag(encoder_name, num_files, frequency, win_length, hop_length, max_seq_len_sec)

# Load all cached arrays (Z is now in compressed .npz format)
Z = np.load(cache_dir / f"Z__{tag}.npz")['Z']  # ← CHANGED: .npy to .npz and added ['Z']
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

## Why This Change is Necessary
- Representation_Extracting now saves Z as a compressed `.npz` file to avoid "File too large" errors
- Clustering_Business must load from `.npz` format and extract the 'Z' key
- All other metadata files (.npy) remain unchanged

## Next Steps
1. Open Clustering_Business notebook
2. Navigate to Cell 5
3. Update line 24 as shown above
4. Run Cell 5 to verify it loads correctly
5. Continue with the rest of the clustering workflow

## Verification
After updating, Cell 5 should successfully load the compressed latent representations without errors.
