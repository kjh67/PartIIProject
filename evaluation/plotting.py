import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from evaluation.eval_utils import ModelMetrics, get_means_and_confidence, read_reprojection_errors


def plot_violin(data, data_labels, y_label, output_file):
    fig, ax = plt.subplots()
    fig.set_size_inches(len(data_labels), 0.6*len(data_labels))
    tick_positions = np.arange(len(data))
    violins = ax.violinplot(data, tick_positions, showmedians=True, showextrema=True)

    # Format axis labels
    ax.set_ylabel(y_label)
    ax.set_ylim(0)
    ax.set_xticks(tick_positions, data_labels)

    fig.savefig(output_file)
    plt.show()


def plot_metrics(plot_data, data_labels, group_labels, y_axis_label, output_path):
    # Plot_data will be an nxm array, where n is the group size and m is the number of bars per group
    fig, ax = plt.subplots()

    group_positions = np.arange(len(group_labels))
    bar_width = 1 / (len(data_labels) + 1)

    # Add bars for each data item within groups
    for data_index, data_label in enumerate(data_labels):
        offset = bar_width * data_index
        means = plot_data[:,data_index,0]
        conf_lower = means - plot_data[:,data_index,1]
        conf_upper = plot_data[:,data_index,2] - means
        print(conf_lower, conf_upper)
        bars = ax.bar(group_positions + offset, means, bar_width, yerr=(conf_lower, conf_upper), capsize=10, label=data_label)
        ax.bar_label(bars, padding=2)

    # Add chart labels, legend etc
    ax.set_ylabel(y_axis_label)
    ax.set_ylim(0)
    ax.set_xticks(group_positions + bar_width*(len(data_labels)-1)/2, group_labels)
    ax.legend(loc='lower right')

    plt.show()
    fig.savefig(output_path)


def plot_all_metrics(source_dirs, output_dir, data_labels, group_labels):
    """Plots PSNR, SSIM, MSE graphs for the project folders passed"""
    metric_sets = []
    for source_location in source_dirs:
        metric_sets += ModelMetrics.load_metrics(os.path.join(source_location, "results", "render_metrics"))

    num_groups = len(group_labels)
    psnr_to_plot = np.array([[get_means_and_confidence(data.psnrs)] for data in metric_sets]).reshape((num_groups, int(len(metric_sets)/num_groups), 3))
    ssim_to_plot = np.array([[get_means_and_confidence(data.ssims)] for data in metric_sets]).reshape((num_groups, int(len(metric_sets)/num_groups), 3))
    mse_to_plot = np.array([[get_means_and_confidence(data.mses)] for data in metric_sets]).reshape((num_groups, int(len(metric_sets)/num_groups), 3))
    
    plot_metrics(psnr_to_plot, data_labels, group_labels, "Peak Signal-to-Noise Ratio", os.path.join(output_dir, "psnrs.png"))
    plot_metrics(ssim_to_plot, data_labels, group_labels, "Structural Similarity", os.path.join(output_dir, "ssims.png"))
    plot_metrics(mse_to_plot, data_labels, group_labels, "Mean Squared Error", os.path.join(output_dir, "mses.png"))


def plot_colmap(source_directories, output_directory, data_labels):
    data = []
    for source_dir in source_directories:
        data.append(read_reprojection_errors(os.path.join(source_dir, "colmap")))
    plot_violin(data, data_labels, "Reprojection Error (px)", os.path.join(output_directory, "colmap_errors.png"))


if __name__ == "__main__":
    parser = ArgumentParser("Script for plotting results of NVS training and COLMAP pose estimation")

    parser.add_argument("mode", choices=["renders", "colmap"])
    parser.add_argument("--source_directories", nargs="+", required=True)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--data_labels", nargs="+")
    parser.add_argument("--group_labels", nargs="+")

    args = parser.parse_args()

    if args.mode == "renders":
        plot_all_metrics(args.source_directories, args.output_directory, args.data_labels, args.group_labels)
    else:
        plot_colmap(args.source_directories, args.output_directory, args.data_labels)
