import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from evaluation.eval_utils import ModelMetrics, get_means_and_confidence, read_reprojection_errors


def plot_violin(data, data_labels, y_label, output_file):

    # Plot the data
    fig, ax = plt.subplots()
    ax.violinplot(data, showmeans=True, showmedians=True)

    # Format axis labels
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel(y_label)

    # Add legend

    fig.savefig(output_file)
    plt.show()


def plot_metrics(plot_data, data_labels, group_labels, y_axis_label, output_path):
    # plot_data will be an nxm array, where n is the group size and m is the number of bars per group
    fig, ax = plt.subplots()

    group_positions = np.arange(len(group_labels))
    bar_width = 1 / (len(data_labels) + 1)
    offset_multiplier = 0

    # Add bars to the figure for each data item
    for group_data in plot_data:
        for data_index, data_item in enumerate(group_data):
            mean, lower_conf, upper_conf = get_means_and_confidence(data_item, 0.95)
            print(mean, lower_conf, upper_conf)
            databar = ax.bar(offset_multiplier*bar_width, mean, bar_width, xerr=mean-lower_conf, yerr=upper_conf-mean, label=data_labels[data_index])
            offset_multiplier += 1
        offset_multiplier += 1

    # Add chart labels, legend etc
    ax.set_ylabel(y_axis_label)
    ax.set_xticks(group_positions+bar_width, group_labels)
    ax.legend(loc='upper right')

    plt.show()
    fig.savefig(output_path)


def plot_all_metrics(source_dirs, output_dir, data_labels, group_labels):
    """Plots PSNR, SSIM, MSE graphs for the project folders passed"""
    metric_sets = []
    for source_location in source_dirs:
        metric_sets += ModelMetrics.load_metrics(os.path.join(source_location, "results", "render_metrics"))

    num_groups = len(group_labels)
    num_datapoints = len(metric_sets[0].psnrs)
    psnrs = np.array([[data.psnrs] for data in metric_sets]).reshape((num_groups, int(len(metric_sets)/num_groups), num_datapoints))
    ssims = np.array([[data.ssims] for data in metric_sets]).reshape((num_groups, int(len(metric_sets)/num_groups), num_datapoints))
    mses = np.array([[data.mses] for data in metric_sets]).reshape((num_groups, int(len(metric_sets)/num_groups), num_datapoints))
    
    plot_metrics(psnrs, data_labels, group_labels, "PSNR", os.path.join(output_dir, "psnrs.png"))
    plot_metrics(ssims, data_labels, group_labels, "SSIM", os.path.join(output_dir, "ssims.png"))
    plot_metrics(mses, data_labels, group_labels, "MSE", os.path.join(output_dir, "mses.png"))


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
