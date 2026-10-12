from pathlib import Path

import h5py
import numpy as np

from .config import DEFAULT_FS, raw_data_dir


def _mat_cell_array(mat, ds, seg_idx):
    """Read one Subj_Wins cell as an array.

    MATLAB cell arrays store either HDF5 references (VITAL/MIMIC PulseDB
    layout, signals live elsewhere in the file) or inline arrays (Aurora
    layout).  Handle both transparently.
    """
    val = ds[0, seg_idx]
    if isinstance(val, np.ndarray):
        return val.flatten()
    return mat[val][:].flatten()


def _mat_cell_scalar(mat, ds, seg_idx):
    """Read one Subj_Wins cell as a scalar (BP labels).

    Aurora stores labels inline as plain float arrays (value per cell);
    PulseDB stores references to scalar datasets.
    """
    val = ds[0, seg_idx]
    if isinstance(val, np.ndarray):
        arr = val.flatten()
        return float(arr[0]) if len(arr) else None
    if isinstance(val, np.generic):  # inline numeric cell (Aurora)
        return float(val)
    return mat[val][()][0, 0]


def get_data_from_mat(mat_path, seg_idx=0, use_filtered=True):
    """Read PPG, ABP, ECG signals and BP labels from a PulseDB .mat file.

    Args:
        mat_path:    Path to the .mat file.
        seg_idx:     Zero-based segment index.
        use_filtered: If True and PPG_F exists, use the official filtered signal.
                      Falls back to PPG_Raw if PPG_F is unavailable.

    Returns:
        ppg: PPG signal (np.array) — filtered if available, else raw.
        abp: ABP signal (np.array).
        ecg: ECG signal (np.array).
        sbp: Systolic BP label.
        dbp: Diastolic BP label.
        has_filtered: True if PPG_F was used (False if raw fallback).
    """
    with h5py.File(mat_path, 'r') as mat:
        subj = mat['Subj_Wins']

        # Prefer official filtered version (PulseDB preprocessed)
        has_filtered = 'PPG_F' in subj
        ppg_key = 'PPG_F' if (use_filtered and has_filtered) else 'PPG_Raw'
        ppg = _mat_cell_array(mat, subj[ppg_key], seg_idx)
        abp = _mat_cell_array(mat, subj['ABP_Raw'], seg_idx)
        ecg = _mat_cell_array(mat, subj['ECG_Raw'], seg_idx)
        sbp = _mat_cell_scalar(mat, subj['SegSBP'], seg_idx)
        dbp = _mat_cell_scalar(mat, subj['SegDBP'], seg_idx)

        # ECG R-peaks if available
        rpeaks = None
        try:
            if 'ECG_RPeaks' in subj:
                rpeaks = _mat_cell_array(mat, subj['ECG_RPeaks'], seg_idx).astype(int)
        except Exception:
            rpeaks = None

        # Age (per-subject)
        age = None
        if 'Age' in subj:
            a_ref = subj['Age']
            try:
                if a_ref.dtype == np.dtype('object'):
                    a_data = mat[a_ref[0, 0]][:].flatten()
                    age = float(a_data[0]) if len(a_data) > 0 else None
                else:
                    age = float(a_ref[0, 0])
            except Exception:
                pass

    return ppg, abp, ecg, sbp, dbp, has_filtered, rpeaks, age


def load_pulse_db_signal(source="vital", subject_id="p000001", seg_idx=0):
    """Load a PulseDB PPG segment.

    Args:
        source: Dataset name — 'vital', 'vitaldb', 'mimic', 'mimic3'.
        subject_id: Subject identifier, e.g. 'p000001'.
        seg_idx: Zero-based segment index.

    Returns:
        dict with keys: ppg, abp, ecg, sbp, dbp, fs, source, subject_id, seg_idx.
    """
    data_dir = raw_data_dir(source)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"PulseDB raw directory not found: {data_dir}. "
            "Set PPG_DATA_ROOT to the data root containing <dataset>/raw/ "
            "(default: <DATA_ROOT>/data_v2)."
        )

    if subject_id:
        sample_file = data_dir / f"{subject_id}.mat"
        if not sample_file.exists():
            raise FileNotFoundError(f"Subject file not found: {sample_file}")
    else:
        mat_files = sorted(data_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {data_dir}")
        sample_file = mat_files[0]
        subject_id = Path(sample_file).stem

    ppg, abp, ecg, sbp, dbp, has_filtered, rpeaks, age = get_data_from_mat(sample_file, seg_idx)

    return {
        'ppg': ppg,
        'abp': abp,
        'ecg': ecg,
        'rpeaks': rpeaks,
        'age': age,
        'sbp': sbp,
        'dbp': dbp,
        'fs': DEFAULT_FS,
        'source': source,
        'subject_id': subject_id,
        'seg_idx': seg_idx,
        'has_filtered': has_filtered,
    }


def get_sample_signal():
    """获取示例信号（用于测试）

    Returns:
        dict: 示例信号字典
    """
    try:
        return load_pulse_db_signal(seg_idx=0)
    except Exception as e:
        print(f"Failed to load real data: {e}")
        print("Using synthetic sample data instead")
        fs = DEFAULT_FS
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        ppg = 0.5 + 0.3 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 2.4 * t) + 0.05 * np.random.randn(len(t))
        return {
            'ppg': ppg,
            'abp': None,
            'ecg': None,
            'sbp': 120.0,
            'dbp': 80.0,
            'fs': fs
        }
