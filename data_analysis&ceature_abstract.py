import numpy as np
import os
import nptdms
from scipy.ndimage import label
from scipy.signal import convolve,wiener
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm

window=4096
bl=3
th=5
dt=0.02e-3
folder_path =fr"D:\5-mcdC-5.22-select"
############################################################
############################################################
############################################################
############################################################
############################################################
def find_peaks(data):
    std_data = np.std(data)
    for index, element in enumerate(data):
        if element <= 0:
            data[index] = 0
    start_indices = []
    end_indices = []
    n = len(data)
    i = 0
    # 遍历修正后的数据，找出非零区间的起始和结束索引
    while i < n:
        if data[i] != 0:
            start = i - 1
            while i < n and data[i] != 0:
                i += 1
            end = i + 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    new_arrays = []
    # 根据起始和结束索引，提取非零区间的数据
    for start, end in zip(start_indices, end_indices):
        new_arrays.append(data[start:end])
    new_array2 = []
    # 筛选出非空且最大值大于等于阈值的数组
    for array in new_arrays:
        if any(array) and max(array) >= 10:
            new_array2.append(array)
    length = []
    max_data = []
    avg_data = []
    std_data = []
    result_array = []
    # 计算筛选后数组的长度、最大值和平均值，并存储结果
    for array in new_array2:
        length.append((len(array) - 1)*0.02)
        max_data.append(max(array))
        avg_data.append(sum(array) / len(array))
        std_data.append(np.std(array))
        result_array.append([(len(array) - 1)*0.02,max(array), sum(array) / len(array),np.std(array)])

    return result_array,new_array2
def find_clusters(data):
    std = np.std(data)
    baseline = bl * std
    threshold = th * std
    filtered = np.array(data) - baseline
    peak_signal=np.where(filtered>=threshold,data,0.0)
    # 对峰值信号进行卷积操作
    convolved = convolve(peak_signal,gaussianed, mode='same')
    # 对卷积结果进行标记
    labeled, num = label(convolved >= 0.1)
    clusters = []
    cluster_data = []
    # 遍历标记结果，找出符合条件的簇
    for cid in range(1, num + 1):
        idx = np.where(labeled == cid)
        if len(filtered[idx]) > 1000:
            clusters.append((idx[0][0], idx[0][-1]))
            cluster_data.append(data[idx])
    for i in range(len(cluster_data)):
        # 将标准差添加到簇数据中
        cluster_data[i] = np.concatenate([cluster_data[i],[std]])
    return cluster_data
def find_cluster_features(clusters_data):
    fft_results=[]
    P1_results=[]
    wiener_P1_results=[]
    P1_cepstrum_results=[]
    cluster_features=[]
    for i, datan in enumerate(clusters_data):
        # 提取簇数据
        cluster_data=datan[:-1]
        time= np.arange(len(cluster_data))*dt
        # 提取簇数据所在文件的标准差
        full_cluster_std=datan[-1]
        cluster_peak,peaks=find_peaks(cluster_data)
        cluster_peak=np.array(cluster_peak)
        if len(cluster_peak)>0:

            # 计算簇数据的平均值
            cluster_avg = np.mean(cluster_data)
            # 计算簇数据的最大值
            cluster_max = np.max(cluster_data)
            # 计算簇数据的最小值
            cluster_min = np.min(cluster_data)
            # 计算簇数据的标准差
            cluster_std = np.std(cluster_data)
            # 计算簇数据的长度
            cluster_length = len(cluster_data)
            # 计算簇数据的峰值数量
            peak_num = len(cluster_peak)
            # 计算峰值最大值的平均值
            peak_max_avg = np.mean(cluster_peak[:,1])
            # 计算峰值最大值的标准差
            peak_max_std = np.std(cluster_peak[:,1])
            # 计算峰的粗糙度的平均值
            peak_roughness_avg = np.mean(cluster_peak[:,3])
            # 计算峰宽的平均值
            peak_length_avg = np.mean(cluster_peak[:,0])
            # 计算峰宽的标准差
            peak_length_std = np.std(cluster_peak[:,0])
            # 计算峰频率
            peak_freq = peak_num/cluster_length
            temp_cluster_features=[]
            print(len(temp_cluster_features))
            # 将簇特征添加到临时列表中
            temp_cluster_features.extend([cluster_avg,cluster_max,cluster_min,cluster_std,
                                          cluster_length,peak_num,peak_max_avg,peak_max_std,peak_roughness_avg,
                                          peak_length_avg,peak_length_std,peak_freq])
            print(len(temp_cluster_features))

            if len(cluster_data) >= 4096:
                diff=(len(cluster_data)-4096)/2
                origin_padded = cluster_data[int(diff):int(diff+4096)]
            else:
                diff = 4096 - len(cluster_data)
                left_pad = diff // 2
                right_pad = diff - left_pad
                origin_padded = np.pad(cluster_data, (left_pad, right_pad), mode="constant", constant_values=0)
            fs = 1 / dt  # 采样频率(Hz)
            n = len(origin_padded)
            fft_data = np.fft.rfft(origin_padded)
            freqs = np.fft.rfftfreq(n, d=1 / fs)
            psd = np.abs(fft_data) ** 2 / (n * fs)
            valid_idx = slice(1, -10) if n > 100 else slice(1, None)
            valid_freqs = freqs[valid_idx]
            valid_psd = psd[valid_idx]
            log_freqs = np.log10(valid_freqs)
            log_psd = np.log10(valid_psd)
            p = np.polyfit(log_freqs, log_psd, 1)  # 一次多项式拟合
            one_over_f_model = 10 ** np.polyval(p, np.log10(freqs[1:]))
            corrected_psd = np.copy(psd)
            """被修正后的功率谱"""
            corrected_psd[1:] /= one_over_f_model
            wiener_psd = np.copy(corrected_psd)
            """维纳滤波后的功率谱"""
            wiener_psd = wiener(wiener_psd, mysize=25)

            log_psd_before = np.log(psd + 1e-10)  # 原始PSD的对数
            cepstrum_before = np.fft.rfft(log_psd_before)
            quefrency_before = np.fft.rfftfreq(len(log_psd_before), d=1 / fs)
            log_psd_after = np.log(corrected_psd + 1e-10)  # 修正后PSD的对数
            """被修正后的倒谱"""
            cepstrum_after = np.fft.rfft(log_psd_after)
            quefrency_after = np.fft.rfftfreq(len(log_psd_after), d=1 / fs)
            log_wiener_psd = np.log(wiener_psd + 1e-10)
            """功率谱维纳滤波后的倒谱"""
            cepstrum_after_wiener = np.fft.rfft(log_wiener_psd)

            frequency_avg_features=[]
            first_quater = 0
            last_quater = 0
            low_3 = 0
            min_3 = 0
            high_3 = 0
            for j in range (61):
                start_index = int(j * (len(corrected_psd)/61))
                end_index = int((j+1) * (len(corrected_psd)/61))
                cluster_P1_part=corrected_psd[start_index:end_index]
                cluster_P1_feature=np.sum(cluster_P1_part)
                if j<15:
                    first_quater+=cluster_P1_feature
                    if j <3:
                        low_3+=cluster_P1_feature
                elif j==29 or j==30 or j==31:
                    min_3+=cluster_P1_feature
                elif j>44:
                    last_quater+=cluster_P1_feature
                    if j>58:
                        high_3+=cluster_P1_feature
                frequency_avg_features.append(cluster_P1_feature)
            temp_cluster_features.extend(frequency_avg_features)
            print(len(temp_cluster_features))
            frequency_avg_full_features=[]
            squar_slices = 51
            squar_root= np.sqrt(freqs[-1])
            # 生成平方根切片的索引
            squar_root_indices = np.linspace(0, squar_root, squar_slices + 1, dtype=int)
            for j in range(squar_slices):
                start_index=squar_root_indices[j]**2
                end_index=squar_root_indices[j+1]**2
                cluster_P1_part=corrected_psd[start_index:end_index]
                cluster_P1_full_feature=np.sum(cluster_P1_part)
                frequency_avg_full_features.append(cluster_P1_full_feature)
            temp_cluster_features.extend(frequency_avg_full_features)
            print(len(temp_cluster_features))

            odd_power=0
            even_power=0
            for j in range(len(corrected_psd)):
                if j%2==0:
                    odd_power+=corrected_psd[i]
                else:
                    even_power+=corrected_psd[i]
            balance1=first_quater/last_quater
            balance2=odd_power/even_power

            sum_power=np.sum(corrected_psd)
            temp_cluster_features.extend([low_3,min_3,high_3,balance1,balance2,sum_power])
            print(len(temp_cluster_features))


            cluster_cepstrum_features=[]
            for j in range(61):
                start_index = int(j * (len(cepstrum_after)/61))
                end_index = int((j+1) * (len(cepstrum_after)/61))
                psd_part=corrected_psd[start_index:end_index]
                sum_power=np.sum(psd_part)
                cluster_cepstrum_features.append(sum_power)
            temp_cluster_features.extend(cluster_cepstrum_features)
            print(len(temp_cluster_features))

            cluster_peaks=[]
            for peak in peaks:
                if len(peak) >= 1024:
                    diff = (len(peak) - 1024) / 2
                    peak_padded = peak[int(diff):int(diff + 1024)]
                else:
                    diff = 1024 - len(peak)
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    peak_padded = np.pad(peak, (left_pad, right_pad), mode="constant", constant_values=0)

                fs = 1 / dt  # 采样频率(Hz)
                peak_n = len(peak_padded)
                peak_fft_data = np.fft.rfft(peak_padded)
                peak_freqs = np.fft.rfftfreq(peak_n, d=1 / fs)
                peak_psd = np.abs(peak_fft_data) ** 2 / (n * fs)
                peak_valid_idx = slice(1, -10) if n > 100 else slice(1, None)
                peak_valid_freqs = peak_freqs[peak_valid_idx]
                peak_valid_psd = peak_psd[peak_valid_idx]
                peak_log_freqs = np.log10(peak_valid_freqs)
                peak_log_psd = np.log10(peak_valid_psd)
                peak_p = np.polyfit(peak_log_freqs, peak_log_psd, 1)  # 一次多项式拟合
                peak_one_over_f_model = 10 ** np.polyval(peak_p, np.log10(peak_freqs[1:]))
                peak_corrected_psd = np.copy(peak_psd)
                """被修正后的功率谱"""
                peak_corrected_psd[1:] /= peak_one_over_f_model

                peak_features=[]
                for j in range(10):
                    start_index = int(j * (len(peak_corrected_psd) / 10))
                    end_index = int((j + 1) * (len(peak_corrected_psd) / 10))
                    peak_P1_part = peak_corrected_psd[start_index:end_index]
                    sum_power=np.sum(peak_P1_part)
                    peak_features.append(sum_power)
                cluster_peaks.append(peak_features)
            cluster_peak_features=[]
            for j in range(10):
                temp_data=np.array(cluster_peaks)[:,j]
                mean_power=np.mean(temp_data)
                std_power=np.std(temp_data)
                cluster_peak_features.extend([mean_power,std_power])
            temp_cluster_features.extend(cluster_peak_features)
            print(len(temp_cluster_features))
            cluster_features.append(temp_cluster_features)
            # if i ==3:
            #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            #     plt.rcParams['axes.unicode_minus'] = False
            #     plt.figure(figsize=(12, 15))
            #
            #     plt.subplot(711)
            #     plt.plot(time_file, data,'r-',alpha=0.7,label='原始电流信号')
            #     plt.plot(time_file, wiener_data_35,'b-',alpha=0.5,label='维纳滤波后')
            #     plt.title('原始电流信号')
            #     plt.xlabel('时间 (s)')
            #     plt.ylabel('电流 (pA)')
            #     plt.legend()
            #     plt.grid(True)
            #
            #     plt.subplot(712)
            #     plt.plot(np.arange(n) * dt, origin_padded)
            #     plt.title('簇电流信号(滤波后)')
            #     plt.xlabel('时间 (s)')
            #     plt.ylabel('电流 (pA)')
            #     plt.grid(True)
            #
            #     plt.subplot(725)
            #     plt.loglog(freqs, psd, label='原始功率谱(log)')
            #     plt.loglog(freqs[1:], one_over_f_model, 'r--', label='1/f模型')
            #     plt.title('功率谱密度与1/f噪声模型')
            #     plt.xlabel('频率 (Hz)')
            #     plt.ylabel('PSD (pA²/Hz)')
            #     plt.legend()
            #     plt.grid(True, which='both')
            #
            #     plt.subplot(726)
            #     plt.loglog(freqs, psd, 'b-', alpha=0.5, label='原始功率谱(log)')
            #     plt.loglog(freqs, corrected_psd, 'g-', label='修正后功率谱(log)')
            #     plt.title('1/f修正前后的功率谱对比(log)')
            #     plt.xlabel('频率 (Hz)')
            #     plt.ylabel('PSD (pA²/Hz)')
            #     plt.legend()
            #     plt.grid(True, which='both')
            #
            #     plt.subplot(714)
            #     plt.plot(freqs, wiener_psd, 'b-', label='维纳滤波后PSD')
            #     plt.plot(freqs, corrected_psd, 'g-', alpha=0.5, label='修正后PSD')
            #     plt.title('1/f修正前后的功率谱对比')
            #     plt.xlabel('频率 (Hz)')
            #     plt.ylabel('PSD (pA²/Hz)')
            #     plt.legend()
            #     plt.grid(True, which='both')
            #
            #     plt.subplot(715)
            #     plt.plot(quefrency_before, np.abs(cepstrum_before), 'b-', label='校准前倒谱')
            #     plt.title('校准前倒谱')
            #     plt.xlabel('倒频率 (samples)')
            #     plt.ylabel('幅度')
            #     plt.legend()
            #     plt.grid(True)
            #
            #     plt.subplot(716)
            #     plt.plot(quefrency_after, np.abs(cepstrum_after_wiener), 'g-', label='维纳滤波后')
            #     plt.plot(quefrency_after, np.abs(cepstrum_after), 'r-', alpha=0.5, label='校准后倒谱')
            #     plt.title('校准后倒谱')
            #     plt.xlabel('倒频率 (samples)')
            #     plt.ylabel('幅度')
            #     # plt.xlim(0, 10000)
            #     plt.legend()
            #     plt.grid(True)
            #
            #     plt.tight_layout()
            #     plt.show()
    return cluster_features
def standardize_data(data):
    data = np.array(data)
    new_data=[]
    EPSILON = 1e-8
    for i in range(data.shape[1]):
        temp_data = data[:, i]
        mean = np.mean(temp_data)
        std = np.std(temp_data)
        if std < EPSILON:
            standardized_data = np.zeros_like(temp_data)
        else:
            standardized_data = (temp_data - mean) / std
        new_data.append(standardized_data)
    new_data = np.array(new_data).T
    # print(new_data)
    return new_data
def draw_peaks(peak_data,save_path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 15))
    max_data = []
    avg_data = []
    length = []
    for data in peak_data:
        length.append(data[0])
        max_data.append(data[1])
        avg_data.append(data[2])

    # 设定 bin 的边界
    max_data_bins = np.arange(0, 100, 0.5)
    avg_data_bins = np.arange(0, 100, 0.5)
    length_bins = np.arange(0, 3, 0.04)

    # 创建一个包含 3 个子图的画布
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig_width, fig_height = fig.get_size_inches()
    aspect_ratio = fig_height / fig_width
    new_width = 6000/600 # 计算新的宽度（英寸）
    new_height = new_width * aspect_ratio  # 计算新的高度（英寸）
    fig.set_size_inches(new_width, new_height)

    # 绘制 max_data 的直方图和对数正态分布拟合图
    n, bins, patches = axes[0].hist(max_data, bins=max_data_bins, density=True, alpha=0.4, color='g', label='Histogram',
                                    edgecolor='black')
    shape, loc, scale = lognorm.fit(max_data)
    x = np.linspace(0, 100, 1000)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)

    # 计算对数正态分布拟合的 R² 值
    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_bin_centers = bin_centers[bin_centers > 0]
    valid_n = n[bin_centers > 0]
    y_pred = lognorm.pdf(valid_bin_centers, shape, loc=loc, scale=scale)
    ss_res = np.sum((valid_n - y_pred) ** 2)
    ss_tot = np.sum((valid_n - np.mean(valid_n)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 构建对数正态分布拟合公式字符串
    formula_str = f"$f(x) = \\frac{{1}}{{x \\sigma \\sqrt{{2\\pi}}}} e^{{-\\frac{{(\\ln(x)-\\mu)^2}}{{2\\sigma^2}}}}$ ($\\mu$={loc:.4f}, $\\sigma$={shape:.4f}, scale={scale:.4f})"

    # 归一化
    max_y = max(np.max(n), np.max(pdf))
    n = n / max_y
    pdf = pdf / max_y

    axes[0].plot(x, pdf, '#ac5e74', lw=2, label=f'Log - Normal Fit \n$R^2$={r2:.4f} \n{formula_str})')
    for patch in patches:
        patch.set_height(patch.get_height() / max_y)
    axes[0].set_title('Peak_MAX')
    axes[0].set_xlabel('Current(pA)')
    axes[0].set_xlim(0, 30)
    axes[0].set_ylabel('Normalized Density')
    axes[0].set_ylim(0, 1.2)
    axes[0].legend()

    # 绘制 avg_data 的直方图和对数正态分布拟合图
    n, bins, patches = axes[1].hist(avg_data, bins=avg_data_bins, density=True, alpha=0.4, color='b', label='Histogram',
                                    edgecolor='black')
    shape, loc, scale = lognorm.fit(avg_data)
    x = np.linspace(0, 100, 1000)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)

    # 计算对数正态分布拟合的 R² 值
    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_bin_centers = bin_centers[bin_centers > 0]
    valid_n = n[bin_centers > 0]
    y_pred = lognorm.pdf(valid_bin_centers, shape, loc=loc, scale=scale)
    ss_res = np.sum((valid_n - y_pred) ** 2)
    ss_tot = np.sum((valid_n - np.mean(valid_n)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 构建对数正态分布拟合公式字符串
    formula_str = f"$f(x) = \\frac{{1}}{{x \\sigma \\sqrt{{2\\pi}}}} e^{{-\\frac{{(\\ln(x)-\\mu)^2}}{{2\\sigma^2}}}}$ ($\\mu$={loc:.4f}, $\\sigma$={shape:.4f}, scale={scale:.4f})"

    # 归一化
    max_y = max(np.max(n), np.max(pdf))
    n = n / max_y
    pdf = pdf / max_y

    axes[1].plot(x, pdf, '#ac5e74', lw=2, label=f'Log - Normal Fit \n$R^2$={r2:.4f} \n{formula_str})')
    for patch in patches:
        patch.set_height(patch.get_height() / max_y)
    axes[1].set_title('Peak_AVG')
    axes[1].set_xlabel('Current(pA)')
    axes[1].set_xlim(0, 30)
    axes[1].set_ylabel('Normalized Density')
    axes[1].set_ylim(0, 1.2)
    axes[1].legend()

    # 绘制 length 的直方图和自定义指数拟合图
    n, bins, patches = axes[2].hist(length, bins=length_bins, density=True, alpha=0.5, color='y', label='Histogram',
                                    edgecolor='black')
    shape, loc, scale = lognorm.fit(length)
    x = np.linspace(0, 6, 100)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)

    # 计算对数正态分布拟合的 R² 值
    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_bin_centers = bin_centers[bin_centers > 0]
    valid_n = n[bin_centers > 0]
    y_pred = lognorm.pdf(valid_bin_centers, shape, loc=loc, scale=scale)
    ss_res = np.sum((valid_n - y_pred) ** 2)
    ss_tot = np.sum((valid_n - np.mean(valid_n)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 构建对数正态分布拟合公式字符串
    formula_str = f"$f(x) = \\frac{{1}}{{x \\sigma \\sqrt{{2\\pi}}}} e^{{-\\frac{{(\\ln(x)-\\mu)^2}}{{2\\sigma^2}}}}$ ($\\mu$={loc:.4f}, $\\sigma$={shape:.4f}, scale={scale:.4f})"

    # 归一化
    max_y = max(np.max(n), np.max(pdf))
    n = n / max_y
    pdf = pdf / max_y

    axes[2].plot(x, pdf, '#ac5e74', lw=2, label=f'Log - Normal Fit \n$R^2$={r2:.4f} \n{formula_str})')
    for patch in patches:
        patch.set_height(patch.get_height() / max_y)
    axes[2].set_title(' Width')
    axes[2].set_xlabel('width(mS)')
    axes[2].set_xlim(0, 3)
    axes[2].set_ylabel('Normalized Density')
    axes[2].set_ylim(0, 1.2)
    axes[2].legend()
    plt.tight_layout(h_pad=3)
    plt.savefig(save_path,dpi=600)
    plt.show()
    return '绘图/保存成功'
############################################################
############################################################
############################################################
############################################################
############################################################
total_features=[]
total_peaks=[]
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.tdms'):
            file_path = os.path.join(root, file)
            try_file = nptdms.TdmsFile.read(file_path)
            data = np.concatenate([ch.data * 1000 for group in try_file.groups() for ch in group.channels()])
            time_file = np.linspace(0, len(data) / 50000, len(data))
            sigma=window/6
            x = np.arange(window) - window // 2
            # 生成高斯函数
            gaussianed=np.exp(-(x ** 2) / (2 * sigma ** 2)) / np.sum(np.exp(-(x ** 2) / (2 * sigma ** 2)))
            # 计算数据的中位数
            median = np.median(data)
            # 减去中位数，对数据进行修正
            data=data-median
            # 计算数据的标准差
            wiener_data_35 = wiener(data, mysize=35)
            clusters= find_clusters(wiener_data_35)
            features=find_cluster_features(clusters)
            total_features.extend(features)
            peaks = find_peaks(data)[0]
            total_peaks.extend(peaks)
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
print('##############################')
print('##############################')
print('##############################')
print('##############################')
print(len(total_peaks))
print(len(total_features))
df = pd.DataFrame(total_peaks)
# 去掉 .tdms 后缀
base_name = os.path.splitext(os.path.basename(folder_path))[0]
save_path_xlsx = os.path.join(folder_path, f"{base_name}.xlsx")
save_path_png = save_path_xlsx.replace('.xlsx', '.png')
df.to_excel(save_path_xlsx, index=False,
            sheet_name='簇数据',header=[ '峰宽(mS)','最大值(nA)', '平均值(nA)', '标准差(nA)'],
            engine='openpyxl')
standardized_features=standardize_data(total_features)
draw_peaks(total_peaks,save_path_png)
np.save(folder_path +'-features.npy',standardized_features)


