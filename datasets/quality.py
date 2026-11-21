"""Find corrupted .npy cutouts before training, so they are dropped from the sample list instead of reaching the model.

A file is dropped when it
- cannot be loaded (`load_error`),
- cannot be brought to 4 x 300 x 300 (`bad_shape`),
- has more than CORRUPTION_THRESHOLD of its pixels NaN/inf (`too_many_invalid`), or
- is all zero after that (`all_zero`): it carries no signal.
Files with some invalid pixels, at most CORRUPTION_THRESHOLD of them, are kept; LensDatasetMemoryEfficient inpaints them.

Checking means reading every file once, so the result is cached per data directory in `file_qc.json`, keyed by file
name, size and modification time (a changed file is checked again). The cache stores the measured invalid fraction, not
the verdict, so it stays valid if the threshold changes. If the data directory is not writable the cache goes to
~/.cache/lensdetection/file_qc/ instead. Writes are atomic, so parallel jobs can share a cache.
"""
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TARGET_SHAPE = (4, 300, 300)
CORRUPTION_THRESHOLD = 0.5  # max fraction of NaN/inf pixels that inpainting may repair
CACHE_NAME = "file_qc.json"
CACHE_VERSION = 1
FALLBACK_CACHE_DIR = Path.home() / ".cache" / "lensdetection" / "file_qc"
N_WORKERS = min(8, os.cpu_count() or 1)


def to_target_shape(data, target_shape=TARGET_SHAPE):
    """Bring a cutout to (C, H, W): channels-last is transposed, a flat array of the right size reshaped, other spatial
    sizes centre-cropped or zero-padded. Raises ValueError when that is impossible."""
    if data.shape == tuple(target_shape):
        return data
    channels, target_h, target_w = target_shape
    if data.ndim == 3 and data.shape[0] == channels:
        pass
    elif data.ndim == 3 and data.shape[-1] == channels:
        data = data.transpose(2, 0, 1)
    elif data.size == np.prod(target_shape):
        return data.reshape(target_shape)
    else:
        raise ValueError(f"cannot bring shape {data.shape} to {tuple(target_shape)}")
    _, h, w = data.shape
    out = np.zeros(target_shape, dtype=data.dtype)
    h_use, w_use = min(h, target_h), min(w, target_w)
    h_off, w_off = (target_h - h_use) // 2, (target_w - w_use) // 2
    out[:, h_off:h_off + h_use, w_off:w_off + w_use] = data[:, :h_use, :w_use]
    return out


def measure_file(path):
    """{'error': None | 'load_error: ...' | 'bad_shape: ...', 'invalid_frac': float, 'all_zero': bool} for one file."""
    try:
        data = np.load(path, mmap_mode="r").astype(np.float32)
    except Exception as e:  # unreadable, truncated, not an .npy file, ...
        return {"error": f"load_error: {type(e).__name__}: {e}", "invalid_frac": 0.0, "all_zero": False}
    try:
        data = to_target_shape(data)
    except ValueError as e:
        return {"error": f"bad_shape: {e}", "invalid_frac": 0.0, "all_zero": False}
    finite = np.isfinite(data)
    return {"error": None, "invalid_frac": float(1.0 - finite.mean()),
            "all_zero": not bool(np.any(data[finite]))}


def verdict(record, corruption_threshold=CORRUPTION_THRESHOLD):
    """Reason to drop the file, or None to keep it."""
    if record["error"]:
        return record["error"]
    if record["invalid_frac"] > corruption_threshold:
        return f"too_many_invalid: {100 * record['invalid_frac']:.2f}% NaN/inf pixels (> {100 * corruption_threshold:g}%)"
    if record["all_zero"]:
        return "all_zero: no non-zero finite pixel"
    return None


def _cache_path(directory):
    own = Path(directory) / CACHE_NAME
    if os.access(directory, os.W_OK) and (not own.exists() or os.access(own, os.W_OK)):
        return own
    digest = hashlib.sha1(str(Path(directory).resolve()).encode()).hexdigest()[:16]
    return FALLBACK_CACHE_DIR / f"{digest}.json"


def _read_cache(path):
    try:
        with open(path) as f:
            cache = json.load(f)
        if cache.get("version") == CACHE_VERSION:
            return cache.get("files", {})
    except (OSError, ValueError):
        pass
    return {}


def _write_cache(path, directory, files, logger=None):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".file_qc.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"version": CACHE_VERSION, "directory": str(Path(directory).resolve()), "files": files}, f)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates 0600; let the group share the cache
        os.replace(tmp, path)
    except OSError as e:
        if logger:
            logger.warning(f"Could not write the file check cache {path}: {e}")


def check_directory(directory, names, logger=None):
    """{name: record} for the given file names in `directory`, from the cache where size and mtime still match."""
    cache_path = _cache_path(directory)
    cache = _read_cache(cache_path)
    records, todo = {}, []
    for name in names:
        st = os.stat(Path(directory) / name)
        sig = [st.st_size, st.st_mtime_ns]
        hit = cache.get(name)
        if hit and hit.get("sig") == sig:
            records[name] = hit
        else:
            todo.append((name, sig))
    if todo:
        if logger:
            logger.info(f"  checking {len(todo)} of {len(names)} files in {directory} (cache: {cache_path})")
        with ThreadPoolExecutor(N_WORKERS) as pool:
            measured = pool.map(measure_file, [str(Path(directory) / n) for n, _ in todo])
            for (name, sig), record in zip(todo, measured):
                records[name] = {**record, "sig": sig}
        # Re-read before writing so that files another job checked meanwhile are kept.
        _write_cache(cache_path, directory, {**_read_cache(cache_path), **{n: records[n] for n, _ in todo}}, logger)
    return records


def drop_corrupted(samples, corruption_threshold=CORRUPTION_THRESHOLD, logger=None):
    """Drop the corrupted files from a list of Samples (datasets/sources.py).

    Returns (kept_samples, report), where report = {'corruption_threshold', 'per_source': {source: {'checked',
    'dropped', 'to_inpaint', 'dropped_by_reason'}}, 'dropped_files': [{'path', 'source', 'label', 'reason'}]}."""
    by_dir = {}
    for s in samples:
        p = Path(s.path)
        by_dir.setdefault(p.parent, []).append(p.name)
    records = {}
    for directory, names in by_dir.items():
        for name, record in check_directory(directory, names, logger).items():
            records[str(directory / name)] = record

    kept, dropped, per_source = [], [], {}
    for s in samples:
        record = records[str(Path(s.path))]
        reason = verdict(record, corruption_threshold)
        stats = per_source.setdefault(s.source, {"checked": 0, "dropped": 0, "to_inpaint": 0, "dropped_by_reason": {}})
        stats["checked"] += 1
        if reason:
            stats["dropped"] += 1
            kind = reason.split(":")[0]
            stats["dropped_by_reason"][kind] = stats["dropped_by_reason"].get(kind, 0) + 1
            dropped.append({"path": s.path, "source": s.source, "label": int(s.label), "reason": reason})
        else:
            stats["to_inpaint"] += record["invalid_frac"] > 0
            kept.append(s)

    if logger:
        for source, stats in per_source.items():
            by_reason = f" {stats['dropped_by_reason']}" if stats["dropped"] else ""
            logger.info(f"  {source}: dropped {stats['dropped']} corrupted files of {stats['checked']}{by_reason}; "
                        f"{stats['to_inpaint']} kept files have NaN/inf pixels to inpaint")
        for d in dropped[:20]:
            logger.info(f"    dropped {d['path']} ({d['reason']})")
        if len(dropped) > 20:
            logger.info(f"    ... and {len(dropped) - 20} more (see data_quality_report.json)")
    report = {"corruption_threshold": corruption_threshold, "per_source": per_source, "dropped_files": dropped}
    return kept, report


def dropped_counts(report):
    """{source: number of dropped files}, for test_results.json."""
    return {source: stats["dropped"] for source, stats in report["per_source"].items()}


def save_report(report, output_dir):
    """Write the drop report to <output_dir>/data_quality_report.json and return the path."""
    path = Path(output_dir) / "data_quality_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
