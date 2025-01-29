import lumicks.pylake as lk
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
import pywt
from scipy import signal
from scipy import stats
from matplotlib.lines import Line2D


def fill_array_with_zeros(array, target_length):
    if len(array) >= target_length:
        return array
    else:
        padding_length = target_length - len(array)
        padded_array = np.pad(array, (0, padding_length), mode='constant')
        return padded_array
def find_nearest_point(array, point):
    distances = np.abs(array - point)

    nearest_index = np.where(distances == np.min(distances))[0][0]

    return nearest_index
def polyfit(X_arr, y_arr, n):
    z1 = np.polyfit(X_arr, y_arr, n)
    p1 = np.poly1d(z1)
    Z = p1(X_arr)
    return Z
def right_shift(data, n):
    copy1 = list(data[n:])
    copy2 = list(data[:n])
    return copy1 + copy2
def back_shift(data, n):
    p = len(data) - n
    copy1 = list(data[p:])
    copy2 = list(data[:p])
    return copy1 + copy2
def get_var(cD):
    coeffs = cD
    abs_coeffs = []
    for coeff in coeffs:
        abs_coeffs.append(math.fabs(coeff))
    abs_coeffs.sort()
    pos = math.ceil(len(abs_coeffs) / 2)
    var = abs_coeffs[pos] / 0.6745
    return var
def sure_shrink(var, coeffs):
    N = len(coeffs)
    sqr_coeffs = []
    for coeff in coeffs:
        sqr_coeffs.append(math.pow(coeff, 2))
    sqr_coeffs.sort()
    pos = 0
    r = 0
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        new_r = (N - 2 * (idx + 1) + (N - (idx + 1))*sqr_coeff + sum(sqr_coeffs[0:idx+1])) / N
        if r == 0 or r > new_r:
            r = new_r
            pos = idx
    thre = math.sqrt(var) * math.sqrt(sqr_coeffs[pos])
    return thre
def visu_shrink(var, coeffs):
    N = len(coeffs)
    thre = math.sqrt(var) * math.sqrt(2 * math.log(N))
    return thre
def heur_sure(var, coeffs):
    N = len(coeffs)
    s = 0
    for coeff in coeffs:
        s += math.pow(coeff, 2)
    theta = (s - N) / N
    miu = math.pow(math.log2(N), 3/2) / math.pow(N, 1/2)
    if theta < miu:
        return visu_shrink(var, coeffs)
    else:
        return min(visu_shrink(var, coeffs), sure_shrink(var, coeffs))
def mini_max(var, coeffs):
    N = len(coeffs)
    if N > 32:
        return math.sqrt(var) * (0.3936 + 0.1829 * math.log2(N))
    else:
        return 0
def get_baseline(data, wavelets_name='sym8', level=5):
    wave = pywt.Wavelet(wavelets_name)
    coeffs = pywt.wavedec(data, wave, level=level)
    for i in range(1, len(coeffs)):
        coeffs[i] *= 0
    baseline = pywt.waverec(coeffs, wave)
    return baseline
def tsd(data, method='sureshrink', mode='soft', wavelets_name='sym8', level=5):

    methods_dict = {'visushrink': visu_shrink, 'sureshrink': sure_shrink, 'heursure': heur_sure, 'minmax': mini_max}
    wave = pywt.Wavelet(wavelets_name)

    data_ = data[:]

    (cA, cD) = pywt.dwt(data=data_, wavelet=wave)
    var = get_var(cD)

    coeffs = pywt.wavedec(data=data, wavelet=wavelets_name, level=level)

    for idx, coeff in enumerate(coeffs):
        if idx == 0:
            continue
        thre = methods_dict[method](var, coeff)
        coeffs[idx] = pywt.threshold(coeffs[idx], thre, mode=mode)

    thresholded_data = pywt.waverec(coeffs, wavelet=wavelets_name)

    return thresholded_data
def ti(data, step=100, method='heursure', mode='soft', wavelets_name='sym5', level=5):
    num = math.ceil(len(data)/step)
    final_data = [0]*len(data)
    for i in range(num):
        temp_data = right_shift(data, i*step)
        temp_data = tsd(temp_data, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
        temp_data = temp_data.tolist()
        temp_data = back_shift(temp_data, i*step)
        final_data = list(map(lambda x, y: x+y, final_data, temp_data))

    final_data = list(map(lambda x: x/num, final_data))

    return final_data



color_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

num_runs=200
for run in range(num_runs):

    date = ['230308','230308', '230606']

    for date in date:
        force_ploy_data_list = []
        distance_ploy_data_list = []
        force_ploy_noise_data_list = []
        distance_ploy_noise_data_list = []

        folder_path = r'E:/experiment/' + str(date) + '/'
        subfolder_count = 0
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolder_count += 1
        print(f"子文件夹数量：{subfolder_count}")

        for g in range(0, subfolder_count):
            group = g
            directory = r'E:/experiment/' + str(date) + '/' + str(group) + '/'
            files = glob.glob(directory + '*FD Curve*.h5')
            baseline_file_name = '00000000-000000 FD Curve ' + str(g)
            if len(directory) < 17:
                baseline_file_directory = directory[0:19]
                count = directory[7:12]
            else:
                baseline_file_directory = directory[0:21]
                count = directory[7:14]
            baseline_file = lk.File(str(baseline_file_directory) + str(baseline_file_name) + '.h5')

            for file_name in files:
                print(file_name[20:39])
                curve_file = lk.File(file_name)
                curve_force = curve_file.downsampled_force2x.data
                curve_distance = curve_file.distance1.data
                baseline_force = baseline_file.downsampled_force2x.data
                baseline_distance = baseline_file.distance1.data

                time_baseline = np.linspace(0, len(baseline_force), len(baseline_force))
                Z_baseline_force_baseline = polyfit(time_baseline, baseline_force, 3)

                baseline_distance_begin_index = np.argmin(np.abs(baseline_distance - curve_distance[0]))
                baseline_distance_end_index = np.argmin(np.abs(baseline_distance - curve_distance[-1]))
                Z_baseline_force_baseline_cutted = Z_baseline_force_baseline[baseline_distance_begin_index:
                                                                             baseline_distance_begin_index+len(curve_force)
                                                   ]
                curve_force_mean = sum(curve_force[:10]) / 10
                Z_baseline_force_mean = sum(Z_baseline_force_baseline_cutted[:10]) / 10
                mean_proportion = curve_force_mean / Z_baseline_force_mean
                force_corrected = abs(curve_force-Z_baseline_force_baseline_cutted*mean_proportion)
                curve_distance = abs(curve_distance-curve_distance[0])*1000

                force_medfilt1 = signal.medfilt(force_corrected, kernel_size=3)
                force_medfilt = signal.medfilt(force_medfilt1, kernel_size=5)

                force_wave = ti(force_corrected,step=len(force_corrected),
                                method='visushrink', mode='soft', wavelets_name='sym5', level=3)

                distance_wave = ti(curve_distance,step=len(curve_distance),
                                method='minmax', mode='soft', wavelets_name='sym5', level=5)


                baseline_force_medfilt = get_baseline(force_wave, wavelets_name='db4', level=5)
                time = np.linspace(0, len(baseline_force_medfilt), len(baseline_force_medfilt))
                Z_curve_force = polyfit(time, baseline_force_medfilt, 4)
                baseline_curve_distance = get_baseline(distance_wave, wavelets_name='db4', level=5)
                Z_curve_distance = polyfit(time, baseline_curve_distance, 4)
                start_point_force = np.random.uniform(4, 8)
                start_point = find_nearest_point(force_corrected, start_point_force)

                big_noise_force = np.random.uniform(3, 5)
                big_noise_point = find_nearest_point(force_corrected, big_noise_force)

                range_availabel = np.arange(start_point, len(Z_curve_force) - 60)

                point1 = np.random.choice(range_availabel)
                point2 = np.random.choice(range_availabel)
                point3 = np.random.choice(range_availabel)
                point4 = np.random.choice(range_availabel)

                if point1 > point2:
                    point1, point2 = point2, point1
                    if point2 > point3:
                        point2, point3 = point3, point2
                        if point3 > point4:
                            point3, point4 = point4, point3

                forcetension1 = np.random.uniform(0.8, 1.0)
                extension1 = np.random.uniform(5, 10)
                forcetension2 = np.random.uniform(0.8, 1.5)
                extension2 = np.random.uniform(5, 15)
                forcetension3 = np.random.uniform(0.8, 2.0)
                extension3 = np.random.uniform(5, 20)
                forcetension4 = np.random.uniform(0.8, 2.4)
                extension4 = np.random.uniform(5, 30)

                if point2-point1>60 and point3 - point2>60 and point4 - point3>60:
                    force_part_1 = Z_curve_force[0:point1]
                    force_part_2 = Z_curve_force[point1:] - forcetension1
                    force_part_3 = force_part_2[:point2 - point1]
                    force_part_4 = force_part_2[point2 - point1:] - forcetension2
                    force_part_5 = force_part_4[:point3 - point2]
                    force_part_6 = force_part_4[point3 - point2:] - forcetension3
                    force_part_7 = force_part_6[:point4 - point3]
                    force_part_8 = force_part_6[point4 - point3:] - forcetension4

                    distance_part_1 = Z_curve_distance[:point1]
                    distance_part_2 = Z_curve_distance[point1:] + extension1
                    distance_part_3 = distance_part_2[:point2 - point1]
                    distance_part_4 = distance_part_2[point2 - point1:] + extension2
                    distance_part_5 = distance_part_4[:point3 - point2]
                    distance_part_6 = distance_part_4[point3 - point2:] + extension3
                    distance_part_7 = distance_part_6[:point4 - point3]
                    distance_part_8 = distance_part_6[point4 - point3:] + extension4

                    force_poly_2 = np.concatenate(
                        [force_part_1, force_part_3, force_part_5, force_part_7,force_part_8])

                    distance_poly_2 = np.concatenate(
                        [distance_part_1, distance_part_3, distance_part_5, distance_part_7,distance_part_8])

                    diff_force_poly_2 = np.diff(force_poly_2)
                    diff_force_corrected = np.diff(force_corrected)
                    diff_distance_poly_2 = np.diff(distance_poly_2)
                    diff_curve_distance = np.diff(curve_distance)

                    index_max_diff_force_corrected = np.argmin(diff_force_corrected)
                    index_max_diff_force_poly_2 = np.argmin(diff_force_poly_2)

                    mean_diff_force_corrected_big = np.mean(diff_force_corrected[0:big_noise_point])
                    std_diff_force_corrected_big = np.std(diff_force_corrected[0:big_noise_point]) - 0.02

                    mean_diff_curve_distance_big = np.mean(diff_curve_distance[0:big_noise_point])
                    std_diff_curve_distance_big = np.std(diff_curve_distance[0:big_noise_point]) - 1

                    mean_diff_force_corrected_sm = np.mean(diff_force_corrected[big_noise_point:len(force_corrected)])
                    std_diff_force_corrected_sm = np.std(
                        diff_force_corrected[big_noise_point:len(force_corrected)]) - 0.08

                    mean_diff_curve_distance_sm = np.mean(diff_curve_distance[big_noise_point:len(force_corrected)])
                    std_diff_curve_distance_sm = np.std(diff_curve_distance[big_noise_point:len(force_corrected)]) - 1

                    force_poly_2_noise_1 = force_poly_2[0:big_noise_point] + np.random.normal(
                        mean_diff_force_corrected_big, std_diff_force_corrected_big,
                        size=force_poly_2[0:big_noise_point].shape)
                    distance_poly_2_noise_1 = distance_poly_2[0:big_noise_point] + np.random.normal(
                        mean_diff_curve_distance_big, std_diff_curve_distance_big,
                         size=distance_poly_2[0:big_noise_point].shape)

                    force_poly_2_noise_2 = force_poly_2[big_noise_point:len(force_corrected)] + np.random.normal(
                        mean_diff_force_corrected_sm, std_diff_force_corrected_sm,
                        size=force_poly_2[big_noise_point:len(force_corrected)].shape)
                    distance_poly_2_noise_2 = distance_poly_2[big_noise_point:len(force_corrected)] + np.random.normal(
                        mean_diff_curve_distance_sm, std_diff_curve_distance_sm,
                        size=distance_poly_2[big_noise_point:len(force_corrected)].shape)

                    force_poly_2_noise = np.concatenate((force_poly_2_noise_1, force_poly_2_noise_2))
                    distance_poly_2_noise = np.concatenate((distance_poly_2_noise_1, distance_poly_2_noise_2))
                    diff_force_medfilt = np.diff(force_corrected)
                    force_noise = diff_force_medfilt - np.mean(diff_force_medfilt)

                    color_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                    plt.figure(figsize=(16, 9))
                    plt.rcParams.update({
                        'font.family': 'Times New Roman',
                        'font.size': 26
                    })

                    plt.plot(distance_poly_2, force_poly_2, linestyle='-', linewidth=4,
                             color=color_hex[3], alpha=0.9, label="Event Location Simulation")
                    plt.plot(distance_poly_2_noise, force_poly_2_noise, linestyle='--', linewidth=4,
                             color=color_hex[5], alpha=0.7, label='Data noise simulation')
                    plt.xlabel('Distance(nm)')
                    plt.ylabel('Force (PN)')
                    custom_legend_entry = Line2D([0], [0], marker='v', color='w', label='Experimental data',
                                                 markerfacecolor='k', markersize=15, linestyle='None')

                    handles, labels = plt.gca().get_legend_handles_labels()

                    handles.append(custom_legend_entry)
                    labels.append("Event")

                    plt.legend(handles=handles, labels=labels)
                    plt.show()

                    diff_curve_distance = np.diff(curve_distance)
                    distance_noise = diff_curve_distance - np.mean(diff_curve_distance)
                    plt.rcParams.update({
                        'font.family': 'Times New Roman',
                        'font.size': 20
                    })

                    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

                    ax[0].hist(force_noise, bins=30, color=color_hex[0], alpha=0.7, label='Force Noise')
                    ax[0].set_xlabel('Noise Magnitude in Force (pN)')
                    ax[0].set_ylabel('Count')
                    ax[0].legend()

                    ax[1].hist(distance_noise, bins=30, color=color_hex[1], alpha=0.7, label='Distance Noise')
                    ax[1].set_xlabel('Noise Magnitude in Distance (nm)')
                    ax[1].set_ylabel('Count')
                    ax[1].legend()

                    plt.tight_layout()
                    plt.show()
                    diff_force_poly_2_noise = np.diff(force_poly_2_noise)
                    noise = diff_force_poly_2_noise - np.mean(diff_force_poly_2_noise)
                    shapiro_test = stats.shapiro(noise)
                    print(f'Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')
                    plt.hist(noise, bins=30)
                    plt.xlabel('force_Noise')
                    plt.ylabel('Frequency')
                    plt.show()

                    diff_distance_poly_2_noise = np.diff(distance_poly_2_noise)
                    noise = diff_distance_poly_2_noise - np.mean(diff_distance_poly_2_noise)
                    shapiro_test = stats.shapiro(noise)
                    print(f'Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')
                    plt.hist(noise, bins=30)
                    plt.xlabel('force_Noise')
                    plt.ylabel('Frequency')
                    plt.show()
                    diff_force_poly_2 = np.diff(force_poly_2)
                    diff_force_corrected = np.diff(force_corrected)
                    diff_distance_poly_2 = np.diff(distance_poly_2)
                    diff_curve_distance = np.diff(curve_distance)

                    color_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                    plt.rcParams.update({
                        'font.family': 'Times New Roman',
                        'font.size': 14
                    })
                    plt.figure(figsize=(7, 3.0))
                    diff_force_corrected = np.diff(force_corrected)
                    noise_force_corrected = diff_force_corrected - np.mean(diff_force_corrected)
                    plt.hist(noise_force_corrected, bins=30, alpha=1, color=color_hex[0])

                    diff_force_poly_2_noise = np.diff(force_poly_2_noise)
                    noise_force_poly_2_noise = diff_force_poly_2_noise - np.mean(diff_force_poly_2_noise)
                    plt.hist(noise_force_poly_2_noise, bins=30, alpha=0.8, color=color_hex[1])
                    plt.xlabel('Force Noise(pN)')
                    plt.ylabel('Frequency')
                    plt.legend(['Experimental', 'Simulated'], loc='upper left')
                    plt.show()

                    plt.rcParams.update({
                        'font.family': 'Times New Roman',
                        'font.size': 14
                    })
                    plt.figure(figsize=(7, 3.0))
                    diff_curve_distance = np.diff(curve_distance)
                    noise_curve_distance = diff_curve_distance - np.mean(diff_curve_distance)
                    plt.hist(noise_curve_distance, bins=30, alpha=1, color=color_hex[2])

                    diff_distance_poly_2_noise = np.diff(distance_poly_2_noise)
                    noise_distance_poly_2_noise = diff_distance_poly_2_noise - np.mean(diff_distance_poly_2_noise)
                    plt.hist(noise_distance_poly_2_noise, bins=30, alpha=0.8, color=color_hex[3])
                    plt.xlabel('Distance Noise(nm)')
                    plt.ylabel('Frequency')
                    plt.legend(['Experimental', 'Simulated'], loc='upper left')
                    plt.show()

                    plt.plot(force_poly_2_noise)
                    plt.show()
                    plt.plot(distance_poly_2_noise)
                    plt.show()
                    plt.plot(distance_poly_2_noise,force_poly_2_noise)
                    plt.show()

                    diff_force_poly_2_noise = np.diff(force_poly_2_noise)
                    noise = diff_force_poly_2_noise - np.mean(diff_force_poly_2_noise)
                    shapiro_test = stats.shapiro(noise)
                    print(f'Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')
                    plt.hist(noise, bins=30)
                    plt.xlabel('force_Noise')
                    plt.ylabel('Frequency')
                    plt.show()

                    diff_distance_poly_2_noise = np.diff(distance_poly_2_noise)
                    noise = diff_distance_poly_2_noise - np.mean(diff_distance_poly_2_noise)
                    shapiro_test = stats.shapiro(noise)
                    print(f'Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')
                    plt.hist(noise, bins=30)
                    plt.xlabel('force_Noise')
                    plt.ylabel('Frequency')
                    plt.show()

                    import numpy as np
                    import scipy.stats as stats

                    experimental_data = noise_force_corrected
                    simulational_data = noise_force_poly_2_noise

                    lower_bound = -0.5
                    upper_bound = 0.5

                    filtered_experimental = experimental_data[
                        (experimental_data >= lower_bound) & (experimental_data <= upper_bound)]
                    filtered_simulational = simulational_data[
                        (simulational_data >= lower_bound) & (simulational_data <= upper_bound)]

                    print("Mean of Filtered Experimental Data:", np.mean(filtered_experimental))
                    print("Standard Deviation of Filtered Experimental Data:", np.std(filtered_experimental))
                    print("Mean of Filtered Simulational Data:", np.mean(filtered_simulational))
                    print("Standard Deviation of Filtered Simulational Data:", np.std(filtered_simulational))

                    ks_stat, ks_p = stats.ks_2samp(filtered_experimental, filtered_simulational)
                    print("KS Statistic:", ks_stat, "P-Value:", ks_p)

                    print("Anderson-Darling Test Result:",
                          stats.anderson_ksamp([filtered_experimental, filtered_simulational]))


                    def bhattacharyya_distance(d1, d2):
                        bc = stats.norm(np.mean(d1), np.std(d1)).pdf(np.linspace(min(d1), max(d1), 1000))
                        bd = stats.norm(np.mean(d2), np.std(d2)).pdf(np.linspace(min(d2), max(d2), 1000))
                        return -np.log(np.sum(np.sqrt(np.multiply(bc, bd))))


                    print("Bhattacharyya Distance:",
                          bhattacharyya_distance(filtered_experimental, filtered_simulational))


                    if force_corrected[-1] > 35:

                        color_hex = ['#dae9fd', '#d77470', '#fbe5cb', '#e1d6e6', '#9467bd',
                                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                        plt.figure()
                        plt.plot(curve_distance, force_corrected, color=color_hex[0], alpha=0.6,
                                 label='Force-Distance Curve')

                        plt.xlabel('Distance(nm)', fontsize=14)
                        plt.ylabel('Force(pN)', fontsize=14)

                        plt.tight_layout()
                        plt.show()

                        plt.figure()
                        plt.rcParams.update({
                            'font.family': 'Times New Roman',
                            'font.size': 14
                        })

                        plt.plot(distance_poly_2_noise, force_poly_2_noise, color=color_hex[0], alpha=0.6)
                        plt.plot(curve_distance, force_corrected, color=color_hex[7], alpha=0.6)
                        plt.plot(Z_curve_distance, Z_curve_force, color=color_hex[3], alpha=0.6)

                        plt.xlabel('Distance(nm)')
                        plt.ylabel('Force(pN)')

                        plt.legend(['Experimental', 'Simulated', 'Fitted Curve'])
                        plt.tight_layout()

                        plt.show()

                        plt.rcParams.update({
                            'font.family': 'Times New Roman',
                            'font.size': 14
                        })
                        plt.figure(figsize=(7, 3.0))
                        diff_force_corrected = np.diff(force_corrected)
                        noise_force_corrected = diff_force_corrected - np.mean(diff_force_corrected)
                        plt.hist(noise_force_corrected, bins=30, alpha=1,color=color_hex[0])
                        diff_force_poly_2_noise = np.diff(force_poly_2_noise)
                        noise_force_poly_2_noise = diff_force_poly_2_noise - np.mean(diff_force_poly_2_noise)
                        plt.hist(noise_force_poly_2_noise, bins=30, alpha=0.8, color=color_hex[1])
                        plt.xlabel('Force Noise(pN)')
                        plt.ylabel('Frequency')
                        plt.legend(['Experimental', 'Simulated'], loc='upper left')
                        plt.show()

                        plt.rcParams.update({
                            'font.family': 'Times New Roman',
                            'font.size': 14
                        })
                        plt.figure(figsize=(7, 3.0))
                        diff_curve_distance = np.diff(curve_distance)
                        noise_curve_distance = diff_curve_distance - np.mean(diff_curve_distance)
                        plt.hist(noise_curve_distance, bins=30, alpha=1, color=color_hex[2])

                        diff_distance_poly_2_noise = np.diff(distance_poly_2_noise)
                        noise_distance_poly_2_noise = diff_distance_poly_2_noise - np.mean(diff_distance_poly_2_noise)
                        plt.hist(noise_distance_poly_2_noise, bins=30, alpha=0.8, color=color_hex[3])
                        plt.xlabel('Distance Noise(nm)')
                        plt.ylabel('Frequency')
                        plt.legend(['Experimental', 'Simulated'], loc='upper left')
                        plt.show()

                    def linear_interpolation(y, new_length):
                        original_length = len(y)
                        x_original = np.linspace(0, 1, original_length)
                        x_new = np.linspace(0, 1, new_length)

                        y_new = np.interp(x_new, x_original, y)
                        return y_new


                    data_len = np.random.randint(600, 650)

                    downsampled_force_1_med = signal.medfilt(linear_interpolation(force_poly_2_noise, data_len),
                                                             kernel_size=3)
                    downsampled_distance_1_med = signal.medfilt(linear_interpolation(distance_poly_2_noise, data_len),
                                                                kernel_size=3)


                    force_poly_2_noise_padding = fill_array_with_zeros(downsampled_force_1_med, 1000)
                    distance_poly_2_noise_padding = fill_array_with_zeros(downsampled_distance_1_med, 1000)


                    force_poly_2_noise_fold = np.concatenate((
                        [4],
                        [force_poly_2[point1-1]],
                        [force_poly_2[point1]],
                        [force_poly_2[point2-1]],
                        [force_poly_2[point2]],
                        [force_poly_2[point3-1]],
                        [force_poly_2[point3]],
                        [force_poly_2[point4-1]],
                        [force_poly_2[point4]],
                        [0],
                        [0],
                        [0],
                        [0],
                        force_poly_2_noise_padding
                    ))

                    distance_poly_2_noise_fold = np.concatenate((
                        [4],
                        [distance_poly_2[point1 - 1]],
                        [distance_poly_2[point1]],
                        [distance_poly_2[point2 - 1]],
                        [distance_poly_2[point2]],
                        [distance_poly_2[point3 - 1]],
                        [distance_poly_2[point3]],
                        [distance_poly_2[point4 - 1]],
                        [distance_poly_2[point4]],
                        [0],
                        [0],
                        [0],
                        [0],
                        distance_poly_2_noise_padding
                    ))


                    force_ploy_noise_data_list.append(force_poly_2_noise_fold)
                    distance_ploy_noise_data_list.append(distance_poly_2_noise_fold)

            print(len(force_ploy_data_list))

            if len(force_ploy_noise_data_list) >0:
                force_ploy_noise_data_vstack = np.vstack(force_ploy_noise_data_list)
                distance_ploy_noise_data_vstack = np.vstack(distance_ploy_noise_data_list)
                print(force_ploy_noise_data_vstack.shape)

                np.save('E:/dataset/simulation_data/transfor/' + str(date) + '_force_ploy_noise_4_1000' + str(
                    run) + '_____.npy',
                        force_ploy_noise_data_vstack)
                np.save(
                    'E:/dataset/simulation_data/transfor/' + str(date) + '_distance_ploy_noise_4_1000' + str(
                        run) + '_____.npy',
                    distance_ploy_noise_data_vstack)