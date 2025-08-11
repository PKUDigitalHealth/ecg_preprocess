import numpy as np
import warnings
import os
import pandas as pd
from tqdm import  tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample
# --- 可选加速包 bottleneck ---
try:
    import bottleneck as bn
    _HAS_BN = True
except ImportError:
    _HAS_BN = False
    warnings.warn("bottleneck not found, falling back to numpy median.")


# =====================================================================
#                           基本工具函数
# =====================================================================

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import warnings

# -- bottleneck (可选) -------------------------------------------------
try:
    import bottleneck as bn
    _HAS_BN = True
except ImportError:
    _HAS_BN = False
    warnings.warn("bottleneck not found, using numpy median fallback.")


# =====================================================================
#                           通用函数
# =====================================================================
def notch50Hz(data, bw=20):
    """50 Hz IIR 陷波；支持 (ch, N) 或 (N,)"""
    if bw == 50:
        B, A = [0.38970, 0.63054, 0.38970], [1, 0.63054, -0.22059]
    elif bw == 20:
        B, A = [0.78140, 1.26434, 0.78140], [1, 1.26434, 0.56281]
    elif bw == 30:
        B, A = [0.67666, 1.09486, 0.67666], [1, 1.09486, 0.35332]
    elif bw == 40:
        B, A = [0.55499, 0.89800, 0.55499], [1, 0.89800, 0.10999]
    else:
        B, A = [0.38970, 0.63054, 0.38970], [1, 0.63054, -0.22059]

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        w1 = w2 = 0.0
        for i in range(data.shape[1]):
            w0 = data[ch, i] - A[1] * w1 - A[2] * w2
            out[ch, i] = B[0] * w0 + B[1] * w1 + B[2] * w2
            w2, w1 = w1, w0
    return out if out.shape[0] > 1 else out[0]


def resample_single(ts, fs_in, fs_out):
    """线性重采样，支持多导联"""
    ts = np.asarray(ts, dtype=float)
    if fs_in == fs_out:
        return ts
    if ts.ndim == 1:
        ts = ts[None, :]

    ch, n_old = ts.shape
    dur = n_old / fs_in
    n_new = int(dur * fs_out)
    x_old = np.linspace(0, dur, n_old, endpoint=False)
    x_new = np.linspace(0, dur, n_new, endpoint=False)
    x_new[-1] = x_old[-1]

    out = np.zeros((ch, n_new))
    for c in range(ch):
        f = interp1d(x_old, ts[c], kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        out[c] = f(x_new)
    return out if out.shape[0] > 1 else out[0]


def fast_sliding_mean(data, radius):
    if radius <= 0:
        return data
    win = 2 * radius + 1
    ker = np.ones(win) / win
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    pad = ((0, 0), (radius, radius))
    padded = np.pad(data, pad, mode='edge')
    out = np.zeros_like(data)
    for c in range(data.shape[0]):
        out[c] = np.convolve(padded[c], ker, mode='valid')
    return out if out.shape[0] > 1 else out[0]


def fast_sliding_median(data, radius):
    if radius <= 0:
        return data
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    win = 2 * radius + 1
    if _HAS_BN:
        out = bn.move_median(data, win, axis=1, min_count=1)
    else:
        out = np.zeros_like(data)
        for c in range(data.shape[0]):
            out[c] = np.array([
                np.median(data[c, max(0, i - radius): i + radius + 1])
                for i in range(data.shape[1])
            ])
    return out if out.shape[0] > 1 else out[0]


def time_varying_threshold(dat, radii, coeffs):
    th = np.zeros_like(dat)
    abs_dat = np.abs(dat)
    for r, c in zip(radii, coeffs):
        th += fast_sliding_median(abs_dat, r) / 0.6745 * c
    return np.clip(th, 0, None)


def threshold_vector(thresh, current):
    return np.where(current >= thresh, 1,
                    np.where(current < thresh / 10, -np.inf, 0))


def expand(arr, radius):
    arr = arr.copy()
    for ch in range(arr.shape[0]):
        idx = np.where(arr[ch] == 1)[0]
        if idx.size == 0:
            continue
        left = np.maximum(0, idx - radius)
        right = np.minimum(arr.shape[1], idx + radius + 1)
        for l, r in zip(left, right):
            arr[ch, l:r] += 1
    return arr


def connect(arr, width):
    arr = arr.copy()
    for ch in range(arr.shape[0]):
        idx = np.where(arr[ch] > 0)[0]
        for i in range(len(idx) - 1):
            if idx[i + 1] - idx[i] < width:
                arr[ch, idx[i]: idx[i + 1] + 1] += 1
    return arr


# =====================================================================
#                     ImmediateAverageDecomposeFilter
# =====================================================================
class ImmediateAverageDecomposeFilter:
    MIN_START_DATA_TIME_QUANTUM = 4
    FIXED_SAMPLE_RATE = 500

    LOW_FREQUENCY_WINDOW_RADIUS = 12
    HIGH_BASE_WINDOW_RADIUS = 4
    THRESHOLD_COEFFICIENTS = [30, 30, 30, 30, 20, -5, -2.5]
    THRESHOLD_RADIUS = [500, 250, 125, 84, 44, 20, 12]
    MAX_THRESHOLD_RADIUS = max(THRESHOLD_RADIUS)

    CASHED_THRESHOLD_LENGTH = 120
    SECOND_THRESHOLD_COEFFICIENTS = [3, 2, 1, 1, 2]
    SECOND_THRESHOLD_RADIUS = [500, 250, 125, 84, 44]
    SECOND_HIGH_FREQUENCY_RADIUS = max(SECOND_THRESHOLD_RADIUS)

    FILTERED_GAIN = 1.5
    SECOND_LOW_FREQUENCY_RADIUS_1 = 4
    SECOND_LOW_FREQUENCY_RADIUS_2 = 12
    SECOND_LOW_FREQUENCY_RADIUS_3 = 4

    def __init__(self, full_data: np.ndarray):
        full_data = np.asarray(full_data, dtype=float)
        if full_data.ndim != 2:
            raise ValueError("data must be 2‑D [channels, time]")
        need_len = self.MIN_START_DATA_TIME_QUANTUM * self.FIXED_SAMPLE_RATE
        if full_data.shape[1] < need_len:
            raise ValueError(f"Need ≥{need_len} pts")

        self.num_channels, self.N = full_data.shape
        self.data = full_data

        # -------------------------------- 一次分解 --------------------------------
        lowF = fast_sliding_mean(self.data, self.LOW_FREQUENCY_WINDOW_RADIUS)
        highF = self.data - lowF
        highB = fast_sliding_mean(highF, self.HIGH_BASE_WINDOW_RADIUS)
        highB1 = highB ** 3

        thresh1 = time_varying_threshold(
            highB1, self.THRESHOLD_RADIUS, self.THRESHOLD_COEFFICIENTS
        )
        mask1 = threshold_vector(thresh1, np.abs(highB1))
        mask1 = connect(expand(mask1, 12), 20)
        first_filtered = np.where(mask1 > 0, lowF + highF, lowF)

        # -------------------------------- 二次分解 ---------------------------------
        low1 = fast_sliding_mean(first_filtered,
                                 self.SECOND_LOW_FREQUENCY_RADIUS_1)
        high1 = first_filtered - low1
        low2 = fast_sliding_mean(low1, self.SECOND_LOW_FREQUENCY_RADIUS_2)
        low3 = low1 - low2
        low4 = fast_sliding_mean(low3, self.SECOND_LOW_FREQUENCY_RADIUS_3)

        thresh2 = time_varying_threshold(
            high1, self.SECOND_THRESHOLD_RADIUS,
            self.SECOND_THRESHOLD_COEFFICIENTS
        )
        high_sel = np.where(thresh2 > np.abs(high1), 0, high1)

        # 结果 (缺尾部几百点)
        self.core = low4 * self.FILTERED_GAIN + low2 + high_sel

        # ------------------------- 补足被裁掉的尾部 --------------------------
        drop = self.N - self.core.shape[1]        # 裁掉点数
        if drop > 0:
            tail = self.core[:, -1:]              # 用最后一个点填补
            self.core = np.hstack([self.core, np.repeat(tail, drop, axis=1)])

    # ------------------------------------------------------------------
    def output(self):
        return self.core


# =====================================================================
#                               外部接口
# =====================================================================
def Filter(signal: np.ndarray, sampleRate: int = 500):
    """
    ECG 滤波入口

    Parameters
    ----------
    signal : ndarray
        shape (12, N) ECG 原始数据
    sampleRate : int
        采样率 (Hz)，默认 500

    Returns
    -------
    ndarray
        滤波后 ECG，与输入长度一致
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError("signal 必须是 shape = (12, N)")

    # ---- 1. 重采样至内部 500 Hz ----
    if sampleRate != ImmediateAverageDecomposeFilter.FIXED_SAMPLE_RATE:
        sig_rs = resample_single(signal, sampleRate,
                                 ImmediateAverageDecomposeFilter.FIXED_SAMPLE_RATE)
        work_fs = ImmediateAverageDecomposeFilter.FIXED_SAMPLE_RATE
    else:
        sig_rs = signal
        work_fs = sampleRate

    # ---- 2. 50 Hz 陷波 ----
    sig_rs = notch50Hz(sig_rs, bw=20)

    # ---- 3. IADF ----
    iadf = ImmediateAverageDecomposeFilter(sig_rs)
    filtered_rs = iadf.output()

    # ---- 4. 若需要恢复原采样率 ----
    if sampleRate != work_fs:
        filtered = resample_single(filtered_rs, work_fs, sampleRate)
    else:
        filtered = filtered_rs

    # ---- 5. 中值滤波去基线漂移 ----
    win = int(0.4 * sampleRate)
    if win % 2 == 0:
        win += 1
    baseline = np.zeros_like(filtered)
    for ch in range(12):
        baseline[ch] = medfilt(filtered[ch], kernel_size=win)
    return filtered - baseline


def filter_bandpass(signal, fs):
    """
    Bandpass filter
    :param signal: 2D numpy array of shape (channels, time)
    :param fs: sampling frequency
    :return: filtered signal
    """
    # Remove power-line interference
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])

    # Simple bandpass filter
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

    # Remove baseline wander
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    filter_ecg = filtered_signal - baseline

    return filter_ecg


def plot_12_leads(signal, name='Original'):
    """
    Plot 12-lead ECG
    :param signal: 2D numpy array of shape (12, time)
    :param name: title suffix
    """
    ecg = signal
    titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.rcParams["axes.grid"] = True
    plt.figure()
    for index in range(12):
        plt.subplot(6, 2, index + 1)
        plt.plot(ecg[index, :], linewidth=1)
        plt.title(f'{titles[index]} {name}')
        plt.axis('on')
    plt.tight_layout()
    plt.show()

# def worker(num):
#     """进程执行的任务"""
#     path = "/data1/1shared/lijun/data/Renmin/ECG数据/ECG_npy/"

#     for filename in tqdm(os.listdir(path)):
#         filepath = os.path.join(path, filename)
#         data = np.load(filepath)
#         data = np.transpose(data)
#         ecg_filtered = Filter(data, sampleRate=500)
#         np.save(f'/data1/1shared/lijun/data/Renmin/ECG数据/ECG_filtered_npy/{filename}', np.array(ecg_filtered))
#         # plt.figure(figsize=(30, 10))

#     print(f"Process {num} is running")

# if __name__ == '__main__':
#     # Example usage
#     # Generate synthetic data for demonstration
#     sample_rate = 500  # Original sampling rate
#     sample = data

#     print(sample.shape)

#     # Plot original data
#     plot_12_leads(sample, 'Original')

#     # Apply the Immediate Average Decompose Filter
#     ecg_filtered = Filter(sample, sampleRate=500)

#     # Plot filtered data
#     plot_12_leads(ecg_filtered, 'Filtered')
    
#     # Apply bandpass filter
#     filtered_bandpass = filter_bandpass(sample, sample_rate)

#     # Plot bandpass filtered data
#     plot_12_leads(filtered_bandpass, 'Bandpass Filtered')

#     print('Processing complete.')
import concurrent.futures

def process_file(filepath, output_dir):
    try:
        filename = os.path.basename(filepath)
        data = np.load(filepath)
        data = np.transpose(data)
        ecg_filtered = Filter(data, sampleRate=500)
        out_path = os.path.join(output_dir, filename)
        np.save(out_path, ecg_filtered)
        return f"{filename} done"
    except Exception as e:
        return f"{filename} failed: {str(e)}"

if __name__ == '__main__':
    # path = "/data1/1shared/lijun/data/Renmin/ECG数据/ECG_npy/"

    # for filename in tqdm(os.listdir(path)):
    #     filepath = os.path.join(path, filename)
    #     data = np.load(filepath)
    #     data = np.transpose(data)
    #     ecg_filtered = Filter(data, sampleRate=500)
    #     np.save(f'/data1/1shared/lijun/data/Renmin/ECG数据/ECG_filtered_npy/{filename}', np.array(ecg_filtered))
    #     # plt.figure(figsize=(30, 10))


    input_dir = "/data1/1shared/lijun/data/Renmin/ECG数据/ECG_npy/"
    output_dir = "/data1/1shared/lijun/data/Renmin/ECG数据/ECG_filtered_npy/"

    file_list = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.npy')
    ]

    max_workers = 10  # 使用所有 CPU 减1（保留系统响应）

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f, output_dir) for f in file_list]

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = f.result()
            print(result)