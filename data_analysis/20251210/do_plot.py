import numpy as np
import matplotlib.pyplot as plt
import os

data_to_load = [
    "arr_of_setpoints",
    "loading_signal",
    "trapped_signal",
    "lost_signal",
    "ratio_signal",
    "ratio_lost"
]

def load_data(timestamp):

    basedir = os.path.dirname(os.path.abspath(__file__))
    basefilename = f"20251017_{timestamp}_"
    filedir = os.path.join(basedir, basefilename)

    data = {}
    for signal in data_to_load:
        filename = f"{basefilename}{signal}"
        filepath = os.path.join(filedir, filename)
        data[signal] = np.genfromtxt(filepath, delimiter=",")
    
    return data

def merge_data(all_data):

    merged_data = {}
    for signal in data_to_load:

        list_of_data = []
        for timestamp in all_data:
            list_of_data.append(all_data[timestamp][signal])

        merged_data[signal] = np.concatenate(list_of_data)

    return merged_data

def plot_data(data, logscale=False):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.scatter(data["arr_of_setpoints"], data["trapped_signal"], marker="x", label="Measured")
    ax1.set_title("Lifetime Measurement")
    ax1.set_ylabel("Trapped Count")

    ax2.scatter(data["arr_of_setpoints"], data["ratio_signal"], marker="x", label="Measured")
    ax2.set_ylabel("Trapped Ratio")
    ax2.set_xlabel("Wait Time (us)")

    if logscale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

if __name__ == "__main__":

    timestamps = [
        #"143010",
        "145340",
        "154710",
        "164653"
    ]

    all_data = {}
    for timestamp in timestamps:
        all_data[timestamp] = load_data(timestamp)
        #plot_data(all_data[timestamp])

    merged_data = merge_data(all_data)
    plot_data(merged_data)
    plot_data(merged_data, logscale=True)
    plt.show()