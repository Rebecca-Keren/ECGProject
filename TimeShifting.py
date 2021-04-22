import os
import math
import numpy as np
import matplotlib.pyplot as plt

ECG_SIGNALS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGSignals")


def get_max_amplitude_in_signal(signal):
    max_amplitude = 0
    for sample in signal:
        if (sample > max_amplitude):
            max_amplitude = sample
    return max_amplitude


def get_values_above_threshold(signal, threshold):
    values = []
    count = 0
    for index, sample in enumerate(signal):
        if (sample > threshold):
            if (count != 0 and values[count - 1] == index - 1 and signal[count - 1] < sample):
                del values[-1]
                count = count - 1
            values.append(index)
            count = count + 1
    return values


def compute_time_shifting(original_signal, output_signal):
    max_amplitude_original = get_max_amplitude_in_signal(original_signal)
    max_amplitude_output = get_max_amplitude_in_signal(output_signal)
    original_threshold = max_amplitude_original / 2
    output_threshold = max_amplitude_output / 2
    print(original_threshold)
    print(output_threshold)
    original_indices_above_threshold = get_values_above_threshold(original_signal, original_threshold)
    output_indices_above_threshold = get_values_above_threshold(output_signal, output_threshold)
    print(original_indices_above_threshold)
    print(output_indices_above_threshold)
    sum_time_differences = 0
    for index in range(len(original_indices_above_threshold)):
        print(original_indices_above_threshold[index])
        print(output_indices_above_threshold[index])
        sum_time_differences = sum_time_differences + abs(
            original_indices_above_threshold[index] - output_indices_above_threshold[index])
    return sum_time_differences / len(original_indices_above_threshold)


if __name__ == "__main__":
    for filename in os.listdir(ECG_SIGNALS):
        if "fecg" in filename:
            path = os.path.join(ECG_SIGNALS, filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            path_label = os.path.join(ECG_SIGNALS, "label_f" + end_path)
            original_signal = np.load(path)
            output_signal = np.load(path_label)
            avg_time_shifting = compute_time_shifting(original_signal, output_signal)
            plt.plot(original_signal)
            plt.show()
            plt.close()
            plt.plot(output_signal)
            plt.show()
            plt.close()
            print(avg_time_shifting)