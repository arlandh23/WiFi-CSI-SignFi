"""
Pipeline preprocessing WiFi CSI untuk dataset SignFi (Intel 5300, 3 antena x 30 subcarrier).

Tahapan:
 1. Extraction         : matriks CSI kompleks H -> amplitudo |H| dan fase angle(H)
 2. Noise filtering    : DWT denoising (soft threshold) + Butterworth low-pass zero-phase (sumbu waktu)
 3. Phase sanitization : unwrapping lintas subcarrier + transformasi linier (hapus slope STO/SFO & offset CFO)
 4. Feature structuring: tensor (N, C=2, Subcarrier=90, Time) dan spektrogram PCA-STFT
"""
from collections import Counter

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, stft

STEP_LOG = Counter()   # audit: berapa kali tiap tahap dipanggil

SIGNFI_SUBCARRIER_IDX = -58.0 + 4.0 * np.arange(30)   # sesuai kode referensi SignFi (csi_phase_offset.py)

VARIANTS = ("full", "no_filter", "no_sanitize", "raw")


# ----------------------------------------------------------------- I/O
def load_signfi_mat(path, prefer="csid"):
    """Baca file .mat SignFi (v5 atau v7.3/HDF5) -> csi (N, T, ant, sub) complex64, label (N,) mulai 0."""
    try:
        from scipy.io import loadmat
        items = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:   # MAT v7.3
        import h5py
        items = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                d = f[k][()]
                if getattr(d.dtype, "names", None) and "real" in d.dtype.names:
                    d = d["real"] + 1j * d["imag"]
                items[k] = d
    cands = [k for k, v in items.items() if np.iscomplexobj(v) and np.ndim(v) == 4]
    if not cands:
        raise ValueError(f"Tidak ada array CSI kompleks 4D di {path}. Keys: {list(items)}")
    csi_key = next((k for k in cands if k.lower().startswith(prefer)), cands[0])
    lab_key = next(k for k in items if k.lower().startswith("label"))
    X = np.asarray(items[csi_key])
    y = np.asarray(items[lab_key]).astype(np.int64).ravel()
    shape = list(X.shape)
    ax_n = shape.index(len(y))
    rest = [i for i in range(4) if i != ax_n]
    ax_sub = next(i for i in rest if shape[i] == 30)
    ax_ant = next(i for i in rest if shape[i] == 3 and i != ax_sub)
    ax_t = next(i for i in rest if i not in (ax_sub, ax_ant))
    X = np.ascontiguousarray(np.transpose(X, (ax_n, ax_t, ax_ant, ax_sub)).astype(np.complex64))
    return X, y - y.min(), csi_key, lab_key


# ----------------------------------------------------------------- Tahap 1
def extract_amplitude_phase(H):
    """Tahap 1 - Extraction: amplitudo & fase dari matriks CSI kompleks."""
    STEP_LOG["1_extraction_amp_phase"] += 1
    return np.abs(H).astype(np.float32), np.angle(H).astype(np.float32)


# ----------------------------------------------------------------- Tahap 2
def dwt_denoise(x, wavelet="db4", level=2, axis=1):
    """Tahap 2a - DWT denoising sepanjang sumbu waktu, VisuShrink soft-threshold.
    sigma noise diestimasi dari koefisien detail level terhalus (MAD/0.6745) per deret."""
    STEP_LOG["2a_dwt_denoise"] += 1
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, 0)
    shp = x.shape; T = shp[0]
    x2 = x.reshape(T, -1)
    coeffs = pywt.wavedec(x2, wavelet, level=level, axis=0)
    sigma = np.median(np.abs(coeffs[-1]), axis=0) / 0.6745
    thr = sigma * np.sqrt(2.0 * np.log(T))
    den = [coeffs[0]] + [np.sign(c) * np.maximum(np.abs(c) - thr, 0.0) for c in coeffs[1:]]
    rec = pywt.waverec(den, wavelet, axis=0)[:T].reshape(shp)
    return np.moveaxis(rec, 0, axis).astype(np.float32)


def butterworth_lowpass(x, fs, cutoff, order=4, axis=1):
    """Tahap 2b - Butterworth low-pass zero-phase (filtfilt) sepanjang sumbu waktu."""
    STEP_LOG["2b_butterworth_lowpass"] += 1
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x, axis=axis).astype(np.float32)


def denoise(x, cfg, axis=1):
    """DWT (buang noise broadband/spike) -> Butterworth LPF (buang komponen > cutoff)."""
    x = dwt_denoise(x, cfg["DWT_WAVELET"], cfg["DWT_LEVEL"], axis=axis)
    return butterworth_lowpass(x, cfg["FS"], cfg["LPF_CUTOFF"], cfg["LPF_ORDER"], axis=axis)


# ----------------------------------------------------------------- Tahap 3
def sanitize_phase(phase, k=SIGNFI_SUBCARRIER_IDX):
    """Tahap 3 - Phase sanitization. phase: [..., n_sub] (radian, wrapped).
    (a) unwrap lintas subcarrier, (b) transformasi linier: hapus slope a & offset b."""
    STEP_LOG["3a_phase_unwrap"] += 1
    unw = np.unwrap(phase.astype(np.float64), axis=-1)
    STEP_LOG["3b_phase_linear_transform"] += 1
    a = (unw[..., -1] - unw[..., 0]) / (k[-1] - k[0])
    b = unw.mean(axis=-1) - a * k.mean()
    return (unw - a[..., None] * k - b[..., None]).astype(np.float32)


# ----------------------------------------------------------------- Tahap 4
def to_tensor(amp, pha):
    """Tahap 4a - amp/pha (N, T, 90) -> (N, Channels=2, Subcarriers=90, Time=T)."""
    STEP_LOG["4a_tensor_structuring"] += 1
    return np.ascontiguousarray(np.stack([amp, pha], axis=1).transpose(0, 1, 3, 2), dtype=np.float32)


def stft_shape(cfg, T):
    hop = cfg["STFT_NPERSEG"] - cfg["STFT_NOVERLAP"]
    return cfg["STFT_NPERSEG"] // 2 + 1, (T - cfg["STFT_NPERSEG"]) // hop + 1


def stft_spectrogram(X, cfg):
    """Tahap 4b - per kanal (amp/fase): PCA lintas 90 subcarrier -> N_PCA komponen -> STFT -> log-magnitude.
    X (N, 2, 90, T) -> (N, 2*N_PCA, F, T_frames)."""
    STEP_LOG["4b_pca_stft_spectrogram"] += 1
    k = cfg["N_PCA"]; out = []
    for c in range(X.shape[1]):
        M = X[:, c].transpose(0, 2, 1).astype(np.float64)          # (N, T, 90)
        M = M - M.mean(axis=1, keepdims=True)                      # buang komponen statis
        _, _, Vt = np.linalg.svd(M, full_matrices=False)           # PCA per sampel (batched)
        pcs = np.einsum("ntf,nkf->nkt", M, Vt[:, :k])              # (N, k, T)
        _, _, Z = stft(pcs, fs=cfg["FS"], nperseg=cfg["STFT_NPERSEG"], noverlap=cfg["STFT_NOVERLAP"],
                       boundary=None, padded=False, axis=-1)        # (N, k, F, frames)
        out.append(np.log1p(np.abs(Z)))
    return np.concatenate(out, axis=1).astype(np.float32)


# ----------------------------------------------------------------- Pipeline
def preprocess_batch(H, cfg, variants=("full",)):
    """H (N, T, ant, sub) kompleks -> {variant: tensor (N, 2, ant*sub, T)} (+ 'S' spektrogram untuk 'full').
    'full'        : tahap 1 + 2 + 3 + 4 (pipeline utama)
    'no_filter'   : tanpa tahap 2           -> ablation
    'no_sanitize' : fase mentah (tanpa 3)   -> ablation
    'raw'         : tanpa tahap 2 & 3       -> ablation
    """
    N, T = H.shape[:2]
    amp, pha_raw = extract_amplitude_phase(H)                                 # Tahap 1
    need_san = any(v in ("full", "no_filter") for v in variants)
    need_flt = any(v in ("full", "no_sanitize") for v in variants)
    pha_san = sanitize_phase(pha_raw).reshape(N, T, -1) if need_san else None   # Tahap 3
    amp = amp.reshape(N, T, -1); pha_raw = pha_raw.reshape(N, T, -1)
    amp_f = denoise(amp, cfg) if need_flt else None                          # Tahap 2
    out = {}
    for v in variants:
        if v == "full":
            a, p = amp_f, denoise(pha_san, cfg)
        elif v == "no_filter":
            a, p = amp, pha_san
        elif v == "no_sanitize":
            a, p = amp_f, denoise(pha_raw, cfg)
        elif v == "raw":
            a, p = amp, pha_raw
        else:
            raise ValueError(v)
        out[v] = to_tensor(a, p)                                              # Tahap 4a
    if "full" in out:
        out["S"] = stft_spectrogram(out["full"], cfg)                         # Tahap 4b
    return out


def preprocess_dataset(H, cfg, variants=("full",), chunk=500, verbose=True):
    """Jalankan preprocess_batch per potongan (hemat RAM) lalu gabungkan."""
    parts = {}
    for i in range(0, len(H), chunk):
        o = preprocess_batch(H[i:i + chunk], cfg, variants)
        for k, v in o.items():
            parts.setdefault(k, []).append(v)
        if verbose:
            print(f"\r   diproses {min(i + chunk, len(H))}/{len(H)}", end="")
    if verbose:
        print()
    return {k: np.concatenate(v) for k, v in parts.items()}
