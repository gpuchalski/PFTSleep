# Databricks notebook source
# DBTITLE 1,Pin Zarr 2 (must run BEFORE the pftsleep install)
# MAGIC %pip install "zarr<3" "numcodecs<0.16" -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Install pftsleep
# MAGIC %pip install /Workspace/Users/gpuchalski@kumc.edu/PFTSleep/
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# =============================================================================
# PFTSleep Latent Representation Extraction
# =============================================================================
# PURPOSE
#   Extract per-6-second-window latent embeddings from the PFTSleep foundational
#   transformer (NOT the GRU sleep-stage classifier) for a cohort of clipped
#   SHHS polysomnogram (PSG) recordings stored as Zarr files. The embeddings are
#   written to disk for downstream unsupervised clustering in a separate notebook.
#
# SCOPE / WHAT THIS NOTEBOOK DOES *NOT* DO
#   - It does NOT use scored sleep studies in any way. Scored hypnograms /
#     event files are reserved for post-hoc analysis after clustering and have
#     zero influence on the embeddings produced here.
#   - It does NOT run clustering. Clustering happens in clustering_business_*.
#
# KEY METHODOLOGICAL DECISIONS (see accompanying methods document)
#   1. FIXED 8-HOUR INPUT. The released PFTSleep model accepts ONLY 8-hour
#      (4800 x 6-second-window) input; the authors' README states variable
#      length is not yet supported. Recordings are therefore truncated to 8h
#      or zero-padded to 8h, with a padding mask passed to the model so padded
#      regions are ignored by attention.
#   2. PADDING STRIPPED AFTER EXTRACTION. Embeddings corresponding to padded
#      (non-real) windows are discarded before saving, so zero-padding never
#      reaches the clustering stage and cannot form artifactual clusters.
#   3. CHANNELS CONCATENATED. The transformer's internal representation is
#      [7 channels x 4800 windows x 512]. To preserve how the 7 channels
#      interact within each 6-second window, the 7 per-channel 512-dim vectors
#      are CONCATENATED into a single 3584-dim vector per window (NOT averaged;
#      averaging would destroy cross-channel structure).
#   4. nsrrid TRACEABILITY. Every embedding row carries the 6-digit SHHS
#      nsrrid parsed from its source filename, enabling an exact join back to
#      scored studies during later analysis.
# =============================================================================

# COMMAND ----------

# DBTITLE 1,1. Imports
# PyTorch allocator setting: must be applied BEFORE any torch import. Tells
# the CUDA caching allocator to use expandable memory segments, which is much
# more resilient to fragmentation over long inference runs. This is the
# standard recommendation for multi-hour transformer inference loops and
# materially reduces "tried to allocate N GB" OOM failures late in a run.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import re
import json
import numpy as np
import torch

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pftsleep.transformers import PatchTFTSimple
from pftsleep.slumber import (
    SelfSupervisedTimeFrequencyDataset,
    ALL_FREQUENCY_FILTERS,
    VOLTAGE_CHANNELS,
)

# COMMAND ----------

# DBTITLE 1,2. User-configurable variables
# -----------------------------------------------------------------------------
# INPUT DATA
# -----------------------------------------------------------------------------
# Trimmed (Lights Off -> Lights On) zarrs produced by
# trim_zarrs_to_scored_interval.py. The directory name "_v2" distinguishes
# this from the earlier broken clipped_zarrs (which had ~80-100% NaN on
# slow-rate channels due to a sample-rate bug in EDF->zarr conversion).
zarr_dir  = Path(globals().get("zarr_dir", "/Workspace/Users/gpuchalski@kumc.edu/projects/PFTSleep/clipped_zarrs_v2"))

# 1224 master-CSV rows - 5 subjects whose nsrrid was not in the master CSV
# and were intentionally excluded from the trim pass.
num_files = globals().get("num_files", 50)

# -----------------------------------------------------------------------------
# SIGNAL / WINDOWING  -- these MUST match how PFTSleep was trained.
# -----------------------------------------------------------------------------
frequency       = globals().get("frequency", 125)            # Hz; all channels resampled to 125 Hz
win_length      = globals().get("win_length", 750)            # samples per patch = 6 s * 125 Hz
hop_length      = globals().get("hop_length", 750)            # non-overlapping patches (hop == win)
window_size_sec = win_length / frequency          # = 6.0 seconds

# FIXED 8-HOUR INPUT. Do not change. The released PFTSleep model only accepts
# 8-hour input (4800 windows). See methodological decision #1 above.
max_seq_len_sec = globals().get("max_seq_len_sec", 8 * 3600)                         # 28 800 s
max_seq_len     = max_seq_len_sec * frequency      # 3 600 000 samples
expected_windows = int(max_seq_len_sec / window_size_sec)   # = 4800

# -----------------------------------------------------------------------------
# PERFORMANCE
# -----------------------------------------------------------------------------
# workers: number of DataLoader worker processes. Set to ~ (vCPUs - 2) so the
# main process and CUDA host thread keep a core each. An NC8as_T4_v3 has 8
# vCPUs, hence 6. Oversubscribing (e.g. 16 on 8 cores) adds context-switching
# overhead and does NOT speed loading up.
import os as _os
_avail = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 2)
workers = globals().get("workers", max(0, min(4, _avail - 1)))
print(f"[extract] available CPUs={_avail}, using num_workers={workers}")

# batch_size: number of full 8-hour recordings encoded per GPU forward pass.
# Larger batches keep the T4 busier (less idle time between files) but use
# more VRAM. EMPIRICAL FINDING on this cohort + NC8as_T4_v3 (16 GB VRAM):
# batch_size=2 caused CUDA out-of-memory. Single-file batches are the safe
# default. Raise to 2 ONLY if you have a GPU with >24 GB VRAM, or after
# verifying VRAM headroom with `nvidia-smi` during a test run.
batch_size = globals().get("batch_size", 4)

# use_fp16: run the transformer forward pass in half precision. On a T4 this
# roughly doubles throughput and halves VRAM via tensor cores. It changes
# embedding values very slightly versus FP32 (small rounding differences;
# the downstream clustering is robust to perturbations of this magnitude).
# Set to True for this study; the choice is recorded in the methods document.
# Set back to False if an exact-FP32 reference run is ever needed.
use_fp16 = globals().get("use_fp16", False)

device  = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# MODEL CHECKPOINT
# -----------------------------------------------------------------------------
pft_ckpt_path = Path(
    "/Workspace/Users/gpuchalski@kumc.edu/projects/PFTSleep/models/pft_sleep_encoder.ckpt"
)

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
# All extraction outputs - shards, per-window metadata arrays, and the
# nsrrid_id_map.json - are written under a single cache directory on DBFS.
# Keeping everything in one place makes the cache self-contained: it can be
# copied/zipped/moved as one unit, and the clustering / analysis notebooks
# find every produced artifact via this single path.
cache_dir = Path(globals().get("CACHE_DIR", "/tmp/new_pftsleep_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,3. Channel definitions
# Seven channels, in fixed order. Each inner list holds acceptable channel-name
# aliases across SHHS EDF naming conventions. Order MUST be stable: it defines
# the order of the 7 x 512 blocks in the concatenated 3584-dim embedding.
channels = [
    ["ECG", "ECG (L-R)", "EKG"],                       # 0: cardiac
    ["EOG(L)", "E1", "E1-M1", "EOG-L"],                # 1: left eye movement
    ["EMG", "cchin_1", "chin", "EMG (L-R)"],           # 2: chin muscle tone
    ["EEG", "C3-M2", "C4-M1", "C3-M2", "EEG3"],        # 3: brain
    ["SaO2", "spo2", "SpO2"],                          # 4: oxygen saturation
    ["THOR RES", "thorax", "Thoracic", "Chest", "Thor"],  # 5: thoracic respiration
    ["ABDO RES", "abdomen", "Abdominal", "ABD", "Abdo"],  # 6: abdominal respiration
]
c_in = len(channels)   # = 7

# COMMAND ----------

# DBTITLE 1,4. Load Zarr files and build the nsrrid traceability map
# -----------------------------------------------------------------------------
# Each embedding must be traceable back to its source recording by SHHS nsrrid
# (a 6-digit integer, e.g. 200001). The nsrrid is parsed from the zarr filename.
# -----------------------------------------------------------------------------
NSRRID_REGEX = re.compile(r"(\d{6})")

def extract_nsrrid(path: Path) -> str:
    """Parse the 6-digit SHHS nsrrid out of a zarr filename stem.

    Raises a clear error if no 6-digit token is found, so a misnamed file
    fails loudly rather than silently corrupting traceability.
    """
    m = NSRRID_REGEX.search(path.stem)
    if m is None:
        raise ValueError(
            f"Could not find a 6-digit nsrrid in filename '{path.name}'. "
            f"Every zarr file must contain its SHHS nsrrid."
        )
    return m.group(1)

zarr_files = sorted(zarr_dir.glob("*.zarr"))
if num_files is not None:
    zarr_files = zarr_files[:num_files]

print(f"Found {len(zarr_files)} zarr files in {zarr_dir}")
assert len(zarr_files) > 0, "No Zarr files found."

# v2-trim sanity check: confirm the first zarr looks like a properly-trimmed
# v2 file. trim_zarrs_to_scored_interval.py writes trim_lights_off_sec /
# trim_lights_on_sec as group attrs on every output. If they are missing the
# zarrs are untrimmed (full EDF) and extraction will pad with garbage from
# the un-scored leading/trailing wake periods - failing here is safer.
import zarr as _zarr_check
_probe = _zarr_check.open(str(zarr_files[0]), mode="r")
if "trim_lights_off_sec" not in _probe.attrs:
    raise RuntimeError(
        f"{zarr_files[0].name} does not carry trim_lights_off_sec / trim_lights_on_sec "
        f"attrs - these are written by trim_zarrs_to_scored_interval.py. "
        f"Either the trim step was not run, or zarr_dir points at the wrong directory. "
        f"Expected v2-trimmed zarrs at: {zarr_dir}"
    )
print(f"v2-trim sanity check passed (Lights Off={_probe.attrs['trim_lights_off_sec']}, "
      f"Lights On={_probe.attrs['trim_lights_on_sec']})")
del _probe, _zarr_check

# Parse nsrrid for every file and check uniqueness (a duplicate nsrrid would
# make traceability ambiguous and must be caught before extraction).
zarr_nsrrids = [extract_nsrrid(p) for p in zarr_files]
dupes = {n for n in zarr_nsrrids if zarr_nsrrids.count(n) > 1}
assert not dupes, f"Duplicate nsrrids detected: {sorted(dupes)}"

# Deterministic nsrrid -> integer index (sorted for reproducibility).
sorted_nsrrids   = sorted(zarr_nsrrids)
nsrrid_to_intidx = {nid: i for i, nid in enumerate(sorted_nsrrids)}

# Persist the map so the clustering / analysis notebooks can resolve indices.
with open(cache_dir / "nsrrid_id_map.json", "w") as f:
    json.dump(nsrrid_to_intidx, f, indent=2)
print(f"Wrote nsrrid_id_map.json ({len(nsrrid_to_intidx)} entries) to {cache_dir}")

# COMMAND ----------

# DBTITLE 1,5. Dataset and DataLoader
# -----------------------------------------------------------------------------
# The dataset is configured to emit ONE fixed-length 8-hour sample per file:
#   - max_seq_len_sec / sample_seq_len_sec / sample_stride_sec all = 8h, so a
#     file longer than 8h is truncated and a file shorter than 8h is padded.
#   - return_sequence_padding_mask=True provides the per-window real/padded mask
#     used both by the model (to ignore padding in attention) and by this
#     notebook (to strip padded embeddings before saving).
#   - trim_wake_epochs=False: the input EDFs are ALREADY clipped to
#     Lights Off -> Lights On. Disabling further wake trimming keeps window
#     index w aligned to clipped-recording time (w * 6 s), which preserves
#     clean traceability for the later scored-study analysis.
# -----------------------------------------------------------------------------
print("[extract] constructing dataset (this builds the sample_df index)...", flush=True)
dataset = SelfSupervisedTimeFrequencyDataset(
    zarr_files=zarr_files,
    channels=channels,
    frequency=frequency,
    trim_wake_epochs=False,
    return_hypnogram_every_sec=30,
    hypnogram_frequency=1,
    hypnogram_padding_mask=-100,
    scale_channels=False,
    start_offset_sec=0,
    clip_interpolations=None,
    include_partial_samples=True,
    return_sequence_padding_mask=True,
    butterworth_filters=ALL_FREQUENCY_FILTERS,
    median_filter_kernel_size=3,
    voltage_channels=VOLTAGE_CHANNELS,
    max_seq_len_sec=max_seq_len_sec,
    sample_seq_len_sec=max_seq_len_sec,
    sample_stride_sec=max_seq_len_sec,
)

print("[extract] dataset built; constructing DataLoader...", flush=True)
loader = DataLoader(
    dataset,
    batch_size=batch_size,   # multiple recordings per GPU forward pass
    shuffle=False,           # preserve file order for traceability
    num_workers=workers,
    persistent_workers=(workers > 0),
    pin_memory=True,
    prefetch_factor=(2 if workers > 0 else None),       # each worker stages 2 batches ahead of the GPU
    drop_last=False,         # keep the final partial batch
)
print(f"Dataset + DataLoader ready (batch_size={batch_size}, workers={workers}).", flush=True)

# COMMAND ----------

# DBTITLE 1,6. Build and load the PFTSleep encoder
def build_pftsleep_encoder() -> torch.nn.Module:
    """Construct the PFTSleep transformer encoder.

    All hyperparameters MUST match those used at training time, otherwise the
    checkpoint weights will not load correctly / will not be meaningful.
    """
    return PatchTFTSimple(
        c_in=c_in,
        win_length=win_length,
        hop_length=hop_length,
        max_seq_len=max_seq_len,
        use_revin=True,
        dim1reduce=False,
        affine=True,
        use_flash_attn=False,
        augmentations=["jitter_zero_mask"],
        mask_ratio=0.1,
        n_layers=3,
        d_model=512,
        n_heads=4,
        shared_embedding=False,
        d_ff=2048,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.1,
        act="gelu",
        res_attention=True,
        pre_norm=False,
        store_attn=False,
        pretrain_head=False,
    )

def load_pftsleep_weights(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """Load PFTSleep encoder weights from a Lightning checkpoint.

    The checkpoint stores parameters under a 'model.' prefix; this strips the
    prefix so they map onto the bare encoder module. The model is then set to
    eval() mode (disables dropout) for deterministic inference.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys (expected for non-encoder params)")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys")
    model.eval()
    return model

# Registry pattern keeps the door open for adding other encoders later
# (e.g. SleepFM) without restructuring the extraction loop.
ENCODER_REGISTRY = {
    "PFTSleep": {
        "build": build_pftsleep_encoder,
        "load":  lambda m: load_pftsleep_weights(m, pft_ckpt_path),
    },
}

# COMMAND ----------

# DBTITLE 1,7. Extraction helpers
def cache_tag(encoder_name: str) -> str:
    """Build a deterministic tag describing this extraction configuration.

    The tag is embedded in every output filename so that runs with different
    settings never overwrite each other and are self-documenting.
    """
    return (
        f"{encoder_name}"
        f"__files{len(zarr_files)}"
        f"__freq{frequency}"
        f"__win{win_length}"
        f"__hop{hop_length}"
        f"__max{max_seq_len_sec}"
        f"__concat7ch"          # marks the 3584-dim concatenated-channel layout
    )

def resolve_per_channel_embedding(z_latent: torch.Tensor) -> torch.Tensor:
    """Normalize the encoder output to shape [n_windows, 3584].

    PFTSleep's per-window representation is [7 channels, 4800 windows, 512].
    To preserve cross-channel interaction information, the 7 per-channel
    512-dim vectors are CONCATENATED (not averaged) into one 3584-dim vector
    per window: final layout [n_windows, 7*512] = [n_windows, 3584], with
    channel order matching `channels` above (ECG, EOG, EMG, EEG, SaO2,
    THOR, ABDO).

    Accepts the squeezed (batch dim removed) encoder output.
    """
    if z_latent.ndim == 3:
        # Expected: [7, T, 512]  (channels, windows, feature)
        if z_latent.shape[0] == c_in and z_latent.shape[2] == 512:
            # [7, T, 512] -> [T, 7, 512] -> [T, 3584]
            z = z_latent.permute(1, 0, 2).reshape(z_latent.shape[1], c_in * 512)
            return z
        # Alternative layout some versions emit: [7, 512, T]
        if z_latent.shape[0] == c_in and z_latent.shape[1] == 512:
            # [7, 512, T] -> [T, 7, 512] -> [T, 3584]
            z = z_latent.permute(2, 0, 1).reshape(z_latent.shape[2], c_in * 512)
            return z
        raise ValueError(f"Unexpected 3-D z_latent shape: {tuple(z_latent.shape)}")
    raise ValueError(
        f"Expected a 3-D per-channel z_latent [7, T, 512]; got ndim="
        f"{z_latent.ndim}, shape={tuple(z_latent.shape)}. Concatenating the 7 "
        f"channels requires per-channel output."
    )

def real_window_flags(seq_pad_mask: torch.Tensor, n_windows: int) -> np.ndarray:
    """Convert a per-sample padding mask into a per-window boolean mask.

    PFTSLEEP MASK CONVENTION (important):
      True  = padded position (model should ignore)
      False = real signal

    This is opposite to the intuitive convention. Verified empirically: a
    7.5-hour trimmed file shows 225,000 True ('padded') samples concentrated
    at the start and 3,375,000 False ('real') samples spanning the recording.
    The comparison below selects windows whose majority is FALSE (real).

    The sequence padding mask is at signal-sample resolution (125 Hz). Each
    6-second window spans `hop_length` samples. A window is REAL when the
    majority of its samples are unpadded; padded windows are dropped before
    saving so zero-padding never enters the clustering stage.

    Vectorized: the per-sample mask is reshaped to [n_windows, hop_length] and
    averaged along axis 1 in a single numpy operation. This replaces a
    4800-iteration Python loop that previously ran on the critical path
    between GPU forward passes and left the T4 idle.

    `seq_pad_mask` is the mask for ONE recording (no batch dimension).
    """
    mask_1d = seq_pad_mask.reshape(-1).bool().cpu().numpy()
    needed  = n_windows * hop_length
    if mask_1d.shape[0] < needed:
        # Pad short masks with True (padding) so the reshape is exact; missing
        # samples count as padding under PFTSleep's convention, which is the
        # correct conservative behavior.
        mask_1d = np.concatenate(
            [mask_1d, np.ones(needed - mask_1d.shape[0], dtype=bool)]
        )
    win_view = mask_1d[:needed].reshape(n_windows, hop_length)
    # Real if majority of samples are FALSE (unpadded under PFTSleep convention).
    return win_view.mean(axis=1) <= 0.5

# COMMAND ----------

# DBTITLE 1,8. Extract and cache embeddings
def extract_latents_cached(encoder_name: str, encoder: torch.nn.Module):
    """Run the encoder over every file and cache per-file embedding shards.

    OUTPUTS (all written under /dbfs/tmp/new_pftsleep_cache):
      <tag>_shards/Z_part_NNNNN.npy   per-file embedding matrix [n_real, 3584]
      night_id__<tag>.npy             loader batch index per window (int32)
      time_idx__<tag>.npy             seconds from clip start per window (float32)
      nsrrid__<tag>.npy               6-digit SHHS nsrrid per window (string)
      nsrrid_intidx__<tag>.npy        integer nsrrid index per window (int32)

    Returns a dict bundling the shard directory and the four metadata arrays.
    """
    tag       = cache_tag(encoder_name)
    shard_dir = cache_dir / f"{tag}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    night_id_path       = cache_dir / f"night_id__{tag}.npy"
    time_idx_path       = cache_dir / f"time_idx__{tag}.npy"
    nsrrid_path         = cache_dir / f"nsrrid__{tag}.npy"
    nsrrid_intidx_path  = cache_dir / f"nsrrid_intidx__{tag}.npy"

    # ------------------------------------------------------------------------
    # RESUME-AFTER-OOM SUPPORT
    # ------------------------------------------------------------------------
    # The earlier "metadata exists -> return early" check is removed. Instead:
    #   - Each file's shard AND its per-file metadata (.meta.npz) are written
    #     as a single unit. A shard without its metadata is treated as not done.
    #   - A skipped file (0 real windows) writes a small marker file so we know
    #     not to retry it.
    #   - At the end, all per-file metadata are concatenated into the four big
    #     arrays for the clustering/analysis notebooks to consume.
    # This makes the run idempotent across kernel restarts: if extraction OOMs
    # after file N, restart Python and re-run; files 0..N-1 are skipped instantly.

    def _shard_done(idx: int) -> bool:
        """A file is 'done' when its shard exists AND its per-file metadata
        exists, OR when a 'skipped' marker exists for it. Anything else gets
        reprocessed (including a half-written shard from a crash).

        Metadata is stored as separate .npy files (NOT .npz) because DBFS does
        not support the seek-and-rewrite that .npz/zip finalization requires
        (causes OSError [Errno 5]). meta_nsrrid_intidx is written LAST, so its
        presence is the completion sentinel."""
        shard = shard_dir / f"Z_part_{idx:05d}.npy"
        meta  = shard_dir / f"meta_nsrrid_intidx_{idx:05d}.npy"
        skip  = shard_dir / f"skipped_{idx:05d}.marker"
        return (shard.exists() and meta.exists()) or skip.exists()

    print(f"[{encoder_name}] Extracting embeddings (6-second windows, 3584-dim)...")
    print(f"  batch_size={batch_size}, workers={workers}, "
          f"precision={'FP16' if use_fp16 else 'FP32'}")

    # Count what's already done so the user sees resume progress immediately.
    n_total      = len(zarr_files)
    already_done = sum(1 for i in range(n_total) if _shard_done(i))
    if already_done:
        print(f"  Resume: {already_done}/{n_total} files already complete; "
              f"will process {n_total - already_done} new files.")

    print("[extract] moving encoder to GPU...", flush=True)
    encoder = encoder.to(device)
    encoder.eval()
    print(f"[extract] encoder on {device}; starting encode loop (first batch fetch can take a moment)...", flush=True)

    # Autocast context for the forward pass. When use_fp16 is False this is a
    # no-op nullcontext, so FP32 behavior is exactly unchanged.
    import contextlib
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_fp16 and device == "cuda")
        else contextlib.nullcontext()
    )

    # file_pos tracks the running index of the CURRENT FILE across batches.
    # With batch_size > 1 the loop variable below indexes BATCHES, not files,
    # so the per-file shard number and nsrrid lookup must use file_pos.
    file_pos = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding ({encoder_name})", unit="batch"):

            n_in_batch = batch[0].shape[0]

            # FAST PATH: if every file in this batch is already done, skip the
            # GPU work entirely. This is what makes resume cheap on restart.
            if all(_shard_done(file_pos + b) for b in range(n_in_batch)):
                file_pos += n_in_batch
                continue

            # batch[0]: signal tensor      [B, 7, samples]
            # batch[2]: per-sample mask    [B, samples] (or [B, 1, samples])
            x                = batch[0].to(device, non_blocking=True)
            seq_padding_mask = batch[2].to(device, non_blocking=True)

            # One forward pass for the whole batch (this is the GPU work).
            with autocast_ctx:
                z_batch = encoder(x, sequence_padding_mask=seq_padding_mask)

            # Iterate the files within this batch and write one shard each.
            for b in range(n_in_batch):
                # Even on the slow path, individual files in the batch may be
                # already done; in that case just advance file_pos.
                if _shard_done(file_pos):
                    file_pos += 1
                    continue

                file_nsrrid = zarr_nsrrids[file_pos]

                # Pull this file's slice out of the batch.
                z_latent_b = z_batch[b]              # per-channel, this file
                mask_b     = seq_padding_mask[b]     # per-sample mask, this file

                # Concatenate 7 channels -> [n_windows, 3584].
                z = resolve_per_channel_embedding(z_latent_b)
                n_windows = z.shape[0]

                # Drop padded windows so zero-padding never reaches clustering.
                flags  = real_window_flags(mask_b, n_windows)
                # Cast to float32 first (autocast may leave the tensor in
                # float16); the on-disk shard dtype stays float16 as before.
                z_np   = z.detach().float().cpu().numpy().astype(np.float16)
                z_np   = z_np[flags]
                n_real = z_np.shape[0]

                if n_real == 0:
                    # Write a tiny marker file so the resume logic remembers
                    # we already decided to skip this one.
                    (shard_dir / f"skipped_{file_pos:05d}.marker").touch()
                    print(f"  [{file_pos:04d}] nsrrid {file_nsrrid}: "
                          f"0 real windows after masking - skipped.")
                    file_pos += 1
                    continue

                # First-file sanity print: a full 8-hour file yields ~4800.
                if file_pos == 0:
                    print(f"  First file: {n_windows} total windows, "
                          f"{n_real} real (expected up to {expected_windows}).")

                # Atomic-ish writes: shard first, then each metadata array as a
                # separate .npy. We do NOT use np.savez: .npz is zip-format and
                # requires seek-and-rewrite on close, which fails on DBFS with
                # OSError [Errno 5]. Plain .npy is append-only and safe on DBFS.
                # meta_nsrrid_intidx is written LAST and is the completion
                # sentinel checked by _shard_done; a crash before it leaves the
                # file marked not-done so it is retried on the next run.
                np.save(shard_dir / f"Z_part_{file_pos:05d}.npy", z_np)
                np.save(shard_dir / f"meta_night_id_{file_pos:05d}.npy",
                        np.full(n_real, file_pos, dtype=np.int32))
                np.save(shard_dir / f"meta_time_idx_{file_pos:05d}.npy",
                        np.arange(n_real, dtype=np.float32) * window_size_sec)
                np.save(shard_dir / f"meta_nsrrid_{file_pos:05d}.npy",
                        np.array([file_nsrrid] * n_real, dtype=object),
                        allow_pickle=True)
                np.save(shard_dir / f"meta_nsrrid_intidx_{file_pos:05d}.npy",
                        np.full(n_real, nsrrid_to_intidx[file_nsrrid], dtype=np.int32))

                print(f"  [{file_pos:04d}] nsrrid {file_nsrrid}: {n_real} windows "
                      f"({n_real * window_size_sec / 3600:.2f} h of real signal)")

                file_pos += 1

            # Between batches: release fragmented GPU memory. Inexpensive and
            # delays (though does not fully prevent) cumulative-fragmentation
            # OOM over a multi-hour run.
            if device == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # FINAL CONCATENATION
    # ------------------------------------------------------------------------
    # Gather every per-file metadata file (whether written in this run or a
    # previous one) and concatenate into the four big arrays the downstream
    # notebooks expect. Run order is preserved by sorting on file index.
    # Metadata is stored as separate .npy files (see write block above).
    print(f"\n[{encoder_name}] Concatenating per-file metadata...")
    meta_idxs = sorted({
        int(p.stem.split("_")[-1])
        for p in shard_dir.glob("meta_nsrrid_intidx_*.npy")
    })
    if not meta_idxs:
        raise RuntimeError(
            "No per-file metadata files found. Either no files were processed "
            "in this run, or the shard_dir is wrong."
        )

    all_night_id      = [np.load(shard_dir / f"meta_night_id_{i:05d}.npy")                  for i in meta_idxs]
    all_time_idx      = [np.load(shard_dir / f"meta_time_idx_{i:05d}.npy")                  for i in meta_idxs]
    all_nsrrid        = [np.load(shard_dir / f"meta_nsrrid_{i:05d}.npy", allow_pickle=True) for i in meta_idxs]
    all_nsrrid_intidx = [np.load(shard_dir / f"meta_nsrrid_intidx_{i:05d}.npy")             for i in meta_idxs]

    night_id      = np.concatenate(all_night_id)
    time_idx      = np.concatenate(all_time_idx)
    nsrrid        = np.concatenate(all_nsrrid)
    nsrrid_intidx = np.concatenate(all_nsrrid_intidx)

    np.save(night_id_path,      night_id)
    np.save(time_idx_path,      time_idx)
    np.save(nsrrid_path,        nsrrid)
    np.save(nsrrid_intidx_path, nsrrid_intidx)

    print(f"\n[{encoder_name}] Extraction complete.")
    print(f"  Total real windows : {len(night_id):,}")
    print(f"  Embedding dimension: {c_in * 512} (7 channels x 512, concatenated)")
    print(f"  Files contributing : {len(np.unique(nsrrid)):,}")
    print(f"  Avg windows/file   : {len(night_id) / max(1, len(np.unique(nsrrid))):.0f}")

    return dict(
        tag=tag,
        shard_dir=shard_dir,
        night_id=night_id,
        time_idx=time_idx,
        nsrrid=nsrrid,
        nsrrid_intidx=nsrrid_intidx,
    )

# COMMAND ----------

# DBTITLE 1,9. Run extraction
LATENTS = {}
for enc_name, spec in ENCODER_REGISTRY.items():
    model = spec["build"]()
    model = spec["load"](model)
    LATENTS[enc_name] = extract_latents_cached(enc_name, model)

# Convenience handles for the PFTSleep run.
pft           = LATENTS["PFTSleep"]
shard_dir     = pft["shard_dir"]
night_id      = pft["night_id"]
time_idx      = pft["time_idx"]
nsrrid        = pft["nsrrid"]
nsrrid_intidx = pft["nsrrid_intidx"]

print("\nFinal metadata array shapes:")
for name, arr in [("night_id", night_id), ("time_idx", time_idx),
                  ("nsrrid", nsrrid), ("nsrrid_intidx", nsrrid_intidx)]:
    print(f"  {name:14s}: {arr.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Representation Extraction Complete
# MAGIC
# MAGIC ## What was produced
# MAGIC
# MAGIC Per-6-second-window latent embeddings from the **PFTSleep transformer
# MAGIC encoder** (the GRU sleep-stage classifier was not used). The scored
# MAGIC sleep studies were not consulted at any point and did not influence the
# MAGIC embeddings.
# MAGIC
# MAGIC ## Output files (under `/dbfs/tmp/new_pftsleep_cache`)
# MAGIC
# MAGIC Tag pattern: `PFTSleep__files{N}__freq125__win750__hop750__max28800__concat7ch`
# MAGIC
# MAGIC | File | Contents |
# MAGIC |---|---|
# MAGIC | `{tag}_shards/Z_part_NNNNN.npy` | One embedding matrix per file, shape `[n_real_windows, 3584]`, float16 |
# MAGIC | `night_id__{tag}.npy` | Loader batch index per window (int32) |
# MAGIC | `time_idx__{tag}.npy` | Seconds from clip start (Lights Off) per window (float32) |
# MAGIC | `nsrrid__{tag}.npy` | 6-digit SHHS nsrrid per window (string) |
# MAGIC | `nsrrid_intidx__{tag}.npy` | Integer nsrrid index per window (int32) |
# MAGIC | `nsrrid_id_map.json` | Mapping nsrrid string to integer index |
# MAGIC
# MAGIC ## Embedding layout
# MAGIC
# MAGIC Each row is one 6-second window: a 3584-dim vector formed by
# MAGIC concatenating the 7 per-channel 512-dim transformer outputs in fixed
# MAGIC order (ECG, EOG, EMG, EEG, SaO2, THOR RES, ABDO RES). Concatenation
# MAGIC preserves cross-channel interaction structure; averaging would not.
# MAGIC
# MAGIC ## Tracing an embedding back to a scored study
# MAGIC
# MAGIC For embedding row `i`: `nsrrid[i]` gives the SHHS subject, and
# MAGIC `time_idx[i]` gives the offset in seconds from Lights Off. In the scored
# MAGIC event file, the scored-study clock also starts at Lights Off, so the
# MAGIC corresponding 30-second scored epoch is `floor(time_idx[i] / 30)`.
# MAGIC
# MAGIC ## Next step
# MAGIC
# MAGIC Run the clustering notebook, which loads the shards and metadata above.
