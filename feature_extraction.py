import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis, entropy
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, hilbert, gaussian, convolve, welch
from scipy.fft import fft
import glob
import os

# 高斯包络线函数
def extract_gaussian_envelope(signal, window_size=50):
    analytical_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytical_signal)
    gaussian_window = gaussian(window_size, std=7)
    gaussian_envelope = convolve(amplitude_envelope, gaussian_window, mode='same') / sum(gaussian_window)
    return gaussian_envelope

def closest_pair(peaks, troughs):
    pairs = []
    for peak in peaks:
        closest_trough = troughs[np.argmin(np.abs(troughs - peak))]
        pairs.append((peak, closest_trough))
    return pairs

def extract_hilbert_envelope(signal):
    analytical_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope


def extract_features(signal):
    # 基本和高级统计特征
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    rms = np.sqrt(np.mean(np.square(signal)))

    # 波形特征
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)
    peak_trough_intervals = np.diff(np.concatenate(([0], peaks, troughs, [len(signal)-1])))

    pairs = closest_pair(peaks, troughs)
    peak_to_trough_height_diff = np.abs(signal[np.array(pairs)[:, 0]] - signal[np.array(pairs)[:, 1]])

    # 频域特征
    freqs, power = welch(signal)
    spectral_entropy = entropy(power)
    freq_bandwidth = freqs[np.argmax(power)] - freqs[np.argmin(power)]

    return {
        "mean": mean_val, "std": std_val, "max": max_val, "min": min_val,
        "skewness": skewness, "kurtosis": kurt, "rms": rms,
        "P-T avg": np.mean(peak_trough_intervals),# average peak-to-trough height difference
        "P-T height avg": np.mean(peak_to_trough_height_diff),# Peak-to-trough height difference average
        "spectral_entropy": spectral_entropy, "freq_bandwidth": freq_bandwidth
    }

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d

def extract_robust_envelope(signal, window_size, percentile=95):
    # 初始化包络线数组
    envelope = np.zeros_like(signal)

    # 滑动窗口计算
    for i in range(len(signal)):
        start = max(i - window_size // 2, 0)
        end = min(i + window_size // 2 + 1, len(signal))
        window = signal[start:end]

        # 使用稳健的统计方法（如百分位数）而不是最大值
        envelope[i] = np.percentile(window, percentile)

    # 使用立方插值平滑包络线
    valid_points = ~np.isnan(envelope)
    envelope_interpolator = interp1d(np.arange(len(signal))[valid_points], envelope[valid_points], kind='cubic', fill_value="extrapolate")
    smooth_envelope = envelope_interpolator(np.arange(len(signal)))

    return smooth_envelope

# 示例使用：
window_size = 500  # 根据信号特性调整窗口大小
# smooth_envelope = extract_robust_envelope(your_signal, window_size)




def process_file(file, output_dir):
    df = pd.read_csv(file)
    signal = df.iloc[:, 1]  # 假设第一列是信号
    # signal = extract_gaussian_envelope(signal) #高斯变换
    signal = extract_robust_envelope(signal, window_size) #上包络

    features = extract_features(signal)
    features_df = pd.DataFrame([features])

    # 构建输出文件的路径
    output_file = os.path.join(output_dir, os.path.basename(file).replace('.csv', '_features.csv'))
    features_df.to_csv(output_file, index=False)

# 创建输出目录（如果尚不存在）
output_dir = 'BP_data/SVM_gaussian'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 应用于所有数据文件
file_list = glob.glob('BP_data/test_*.csv')
for file in file_list:
    process_file(file, output_dir)

import matplotlib.pyplot as plt
import pandas as pd

# 设置图表大小和字体样式
plt.figure(figsize=(8, 6))  # 将尺寸转换为厘米
plt.rcParams.update({'font.size': 8*2})  # 设置全局字体大小

# 循环处理每个文件并绘图
for file in file_list:
    df = pd.read_csv(file)
    signal = df.iloc[:, 1]  # 假设第一列是信号
    gaussian_signal = extract_gaussian_envelope(signal)

    plt.plot(gaussian_signal, label=os.path.basename(file))

# 设置标题、轴标签和图例
plt.title('Gaussian Envelopes of Signals', fontsize=10*2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
# plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()




# # 示例：应用上包络函数并绘制
# upper_envelope = extract_upper_envelope(signal)
#
# # 循环处理每个文件并绘图
# for file in file_list:
#     df = pd.read_csv(file)
#     signal = df.iloc[:, 1]  # 假设第一列是信号
#     upper_envelope = extract_upper_envelope(signal)
#     plt.plot(upper_envelope, label=os.path.basename(file))
#
# plt.show()
