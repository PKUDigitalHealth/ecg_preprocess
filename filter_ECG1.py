import numpy as np
import bottleneck as bn
from scipy.interpolate import interp1d
import time
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt, iirnotch, filtfilt, butter
import time           
import json


def timing(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"Function {func.__name__} took {(t2 - t1)*1000:.2f} ms")
        return result
    return wrapper

def notch50Hz(data, bw):
    """ 50Hz陷波器（IIR实现） """
    if bw == 50:
        B = [0.38970, 0.63054, 0.38970]
        A = [1, 0.63054, -0.22059]
    elif bw == 20:
        B = [0.78140, 1.26434, 0.78140]
        A = [1, 1.26434, 0.56281]
    elif bw == 30:
        B = [0.67666, 1.09486, 0.67666]
        A = [1, 1.09486, 0.35332]
    elif bw == 40:
        B = [0.55499, 0.89800, 0.55499]
        A = [1, 0.89800, 0.10999]
    else:
        B = [0.38970, 0.63054, 0.38970]
        A = [1, 0.63054, -0.22059]

    data = np.asarray(data, dtype=float)
    out = np.zeros_like(data)
    w1, w2 = 0.0, 0.0
    for i in range(len(data)):
        w0 = data[i] - A[1] * w1 - A[2] * w2
        out[i] = B[0] * w0 + B[1] * w1 + B[2] * w2
        w2 = w1
        w1 = w0
    return out

def resample_single(ts, fs_in, fs_out):
    ts = np.asarray(ts, dtype=float)
    if fs_out == fs_in:
        return ts
    x_old = np.linspace(0, 1, num=ts.shape[0], endpoint=True)
    x_new = np.linspace(0, 1, num=int(len(ts) * fs_out / fs_in), endpoint=True)
    f = interp1d(x_old, ts, kind='linear')
    return f(x_new)

@timing
def fast_sliding_mean(data, radius):
    """滑动均值，边界自动补齐"""
    data = np.asarray(data)
    window = 2*radius+1
    kernel = np.ones(window) / window
    out = np.convolve(data, kernel, 'same')
    return out

@timing
def fast_sliding_median_bn(data, radius):
    """高效滑动中值，基于bottleneck"""
    window = 2 * radius + 1
    return bn.move_median(data, window, min_count=1)

@timing
def time_varying_threshold_bn(dat, radii, coeffs):
    """多尺度动态阈值，所有窗口用bottleneck并批量加权"""
    dat = np.asarray(dat)
    th_total = np.zeros_like(dat)
    for r, c in zip(radii, coeffs):
        th = fast_sliding_median_bn(np.abs(dat), r)
        th_total += th / 0.6745 * c
    return np.clip(th_total, 0, None)

@timing
def threshold(thresh, current):
    """numpy向量化判别"""
    return np.where(current >= thresh, 1, np.where(current < thresh / 10, -np.inf, 0))

def expand(arr, radius):
    """对arr==1的区间进行扩展，向前后radius内全部+1"""
    arr = np.array(arr, dtype=float)
    idx = np.where(arr == 1)[0]
    if len(idx) == 0:
        return arr
    left = np.maximum(0, idx - radius)
    right = np.minimum(len(arr), idx + radius + 1)
    for l, r in zip(left, right):
        arr[l:r] += 1
    return arr

def connect(arr, width):
    """将arr中相邻>0区段，若间隔小于width，填满为非0"""
    arr = np.array(arr, dtype=float)
    idx = np.where(arr > 0)[0]
    for i in range(len(idx) - 1):
        if idx[i+1] - idx[i] < width:
            arr[idx[i]:idx[i+1]+1] += 1
    return arr

class ImmediateAverageDecomposeFilter:
    MIN_START_DATA_TIME_QUANTUM = 4
    FIXED_SAMPLE_RATE = 125
    LOW_FREQUENCY_WINDOW_RADIUS = 3
    HIGH_BASE_WINDOW_RADIUS = 1
    THRESHOLD_COEFFICIENTS = [30, 30, 30, 30, 20, -5, -2.5]
    THRESHOLD_RADIUS = [125, 63, 31, 21, 11, 5, 3]
    MAX_THRESHOLD_RADIUS = max(THRESHOLD_RADIUS)
    CASHED_THRESHOLD_LENGTH = 30
    SECOND_THRESHOLD_COEFFICIENTS = [3, 2, 1, 1, 2]
    SECOND_THRESHOLD_RADIUS = [125, 63, 31, 21, 11]
    SECOND_HIGH_FREQUENCY_RADIUS = max(SECOND_THRESHOLD_RADIUS)
    FILTERED_GAIN = 1.5
    SECOND_LOW_FREQUENCY_RADIUS_1 = 1
    SECOND_LOW_FREQUENCY_RADIUS_2 = 3
    SECOND_LOW_FREQUENCY_RADIUS_3 = 1

    def __init__(self, start_data):
        if len(start_data) < self.MIN_START_DATA_TIME_QUANTUM * self.FIXED_SAMPLE_RATE:
            raise ValueError(f"At least input {self.MIN_START_DATA_TIME_QUANTUM} seconds of data")
        self.cachedData = np.array(start_data, dtype=float).copy()

        # 多窗口滑动均值/中值
        self.lowFrequencyData = fast_sliding_mean(self.cachedData, self.LOW_FREQUENCY_WINDOW_RADIUS)
        self.highFrequencyData = self.cachedData - self.lowFrequencyData
        self.highBaseData = fast_sliding_mean(self.highFrequencyData, self.HIGH_BASE_WINDOW_RADIUS)
        self.highBaseData1 = self.highBaseData ** 3

        # 主自适应阈值
        thresh = time_varying_threshold_bn(
            self.highBaseData1,
            self.THRESHOLD_RADIUS,
            self.THRESHOLD_COEFFICIENTS
        )
        current = np.abs(self.highBaseData1)
        self.thresholdResult = threshold(thresh, current)

        # 扩展与连接
        self.thresholdResult = expand(self.thresholdResult, 3)
        self.thresholdResult = connect(self.thresholdResult, 5)

        # 一次滤波结果
        self.cachedFirstFilteredData = np.where(self.thresholdResult > 0,
                                                self.lowFrequencyData + self.highFrequencyData,
                                                self.lowFrequencyData)

        # 二次滤波/分解
        self.secondLowFrequencyDat1 = fast_sliding_mean(self.cachedFirstFilteredData, self.SECOND_LOW_FREQUENCY_RADIUS_1)
        self.secondHighFrequencyDat1 = self.cachedFirstFilteredData - self.secondLowFrequencyDat1
        self.secondLowFrequencyDat2 = fast_sliding_mean(self.secondLowFrequencyDat1, self.SECOND_LOW_FREQUENCY_RADIUS_2)
        self.secondLowFrequencyDat3 = self.secondLowFrequencyDat1 - self.secondLowFrequencyDat2
        self.secondLowFrequencyDat4 = fast_sliding_mean(self.secondLowFrequencyDat3, self.SECOND_LOW_FREQUENCY_RADIUS_3)

        # 二次高频阈值
        second_thresh = time_varying_threshold_bn(
            self.secondHighFrequencyDat1,
            self.SECOND_THRESHOLD_RADIUS,
            self.SECOND_THRESHOLD_COEFFICIENTS
        )
        self.secondHighFrequencyDatThresh = second_thresh

    def filter(self, data=None):
        high = np.where(self.secondHighFrequencyDatThresh > np.abs(self.secondHighFrequencyDat1),
                        0, self.secondHighFrequencyDat1)
        callback = (self.secondLowFrequencyDat4 * self.FILTERED_GAIN +
                    self.secondLowFrequencyDat2 + high)
        return callback

@timing
def Filter(signal, sampleRate):
    signal = np.asarray(signal, dtype=float)
    orig_sampleRate = sampleRate

    # 1. 重采样到125Hz
    if sampleRate != 125:
        signal = resample_single(signal, sampleRate, 125)
        sampleRate = 125

    # 2. 陷波50Hz
    signal = notch50Hz(signal, 20)

    # 3. 初始化滤波器并滤波
    IADfilter = ImmediateAverageDecomposeFilter(signal)
    filtered = IADfilter.filter()

    # 4. 如需恢复原采样率
    if orig_sampleRate != 125:
        filtered = resample_single(filtered, 125, orig_sampleRate)

    # 5. 中值滤波去基线漂移（保证奇数窗）
    winlen = int(0.4 * orig_sampleRate)
    if winlen % 2 == 0:
        winlen += 1
    from scipy.signal import medfilt
    baseline = medfilt(filtered, winlen)
    filter_ecg = filtered - baseline
    return filter_ecg

# 带通滤波
def filter_bandpass(signal, fs):
    # remove power-line interference
    b, a = iirnotch(50, 50, fs)
    signal = filtfilt(b, a, signal)

    # simple filter
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    filter_ecg = filtfilt(b, a, signal)

    # 中值滤波去基线漂移
    baseline = medfilt(filter_ecg, int(0.4 * fs) + 1)
    filter_ecg = filter_ecg - baseline

    return filter_ecg

def plot(signal, name='original'):
    # 设置图形的大小
    plt.figure(figsize=(10, 4))
    signal = np.array(signal)
    # 绘制I导联的数据
    plt.plot(signal)
    plt.title('{}'.format(name))
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    # 显示图表
    plt.show()


if __name__ == '__main__':

    path = '/data/0shared/lijun/code/Heartvoice/data/abnormal/813156453924737024.json'
    with open(path, 'r', encoding='utf-8') as file:
      content = file.read()
      data = json.loads(content)
      data = data[0]['ecg_data'] 

    path = '/data/0shared/lijun/code/Heartvoice/data/normal/1043531122752884736.json'
    with open(path, 'r', encoding='utf-8') as file:
      content = file.read()
      data2 = json.loads(content)
      data2 = data2[0]['ecg_data']

    # data = np.loadtxt("/data/0shared/lijun/code/Heartvoice/abnormal_ECG.txt", delimiter=',')
    # data2 = np.loadtxt("/data/0shared/lijun/code/Heartvoice/normal_ECG.txt", delimiter=',')
    # print(data)
    fs = 125
    # plot(data, 'Original')
        
    data_bandpass = filter_bandpass(data, fs)
    # plot(filter2, 'Bandpass')
    start_time = time.time()
    data_Filter = Filter(data, fs)
    end_time = time.time()
    ecg_time = len(data)/125
    print(f"长度: {ecg_time:.1f} 秒")
    print(f"滤波总耗时: {end_time - start_time:.4f} 秒")
    print(f"每秒点数处理: {len(data)/(end_time-start_time):.2f} points/sec")
    # plot(filter1, 'Filter')
    plt.figure(figsize=(30, 10))
    plt.subplot(3, 1, 1)
    plt.plot(data[0:5000], label='Original signal')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(data_Filter[0:5000], label='Filter signal')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(data_bandpass[0:5000], label='Bandpass')
    plt.legend()


    data2_bandpass = filter_bandpass(data2, fs)
    # plot(filter2, 'Bandpass')
    start_time = time.time()
    data2_Filter = Filter(data2, fs)
    end_time = time.time()
    ecg_time = len(data2)/125
    print(f"长度: {ecg_time:.1f} 秒")
    print(f"滤波总耗时: {end_time - start_time:.4f} 秒")
    print(f"每秒点数处理: {len(data2)/(end_time-start_time):.2f} points/sec")
    # plot(filter1, 'Filter')
    plt.figure(figsize=(30, 10))
    plt.subplot(3, 1, 1)
    plt.plot(data2[0:5000], label='Original signal')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(data2_Filter[0:5000], label='Filter signal')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(data2_bandpass[0:5000], label='Bandpass')
    plt.legend()
    plt.show()
