import pandas as pd
import numpy as np
import glob
import pywt


# 使用小波变换的下采样函数
def downsample_data_with_wavelet(file_path, downsample_factor, wavelet='db1', level=1):
    # 读取数据
    df = pd.read_csv(file_path)

    # 处理每个文件时，确保处理逻辑独立
    original_length = len(df)

    # 小波分解
    coeffs = pywt.wavedec(df['Ampl'], wavelet, level=level)

    # 重构信号，只使用近似系数
    reconstructed_signal = pywt.waverec(coeffs[:1] + [None] * len(coeffs[1:]), wavelet)

    # 如果重构信号长度不匹配，进行调整
    if len(reconstructed_signal) > original_length:
        reconstructed_signal = reconstructed_signal[:original_length]
    elif len(reconstructed_signal) < original_length:
        reconstructed_signal = np.pad(reconstructed_signal, (0, original_length - len(reconstructed_signal)),
                                      'constant')

    # 更新信号
    df['Ampl'] = reconstructed_signal

    # 下采样
    downsampled_df = df.iloc[::downsample_factor, :]
    return downsampled_df.iloc[:int(np.ceil(original_length / downsample_factor)), :]  # 确保返回的长度与原始数据一致


# 设置参数
downsample_factor = 4  # 下采样因子

# 查找所有以 'test' 开头的 CSV 文件
file_list = glob.glob('BP_data/test_*.csv')

# 对每个文件应用下采样并保存结果
for file in file_list:
    downsampled_df = downsample_data_with_wavelet(file, downsample_factor)
    downsampled_file_name = file.replace('test', 'downsample/test')
    downsampled_df.to_csv(downsampled_file_name, index=False)
