import matplotlib.pyplot as plt

# plotting utilities


def plot_violin(data, data_labels, y_label, output_file):

    # Plot the data
    fig, ax = plt.subplots()
    ax.violinplot(data, showmeans=True, showmedians=True)

    # Format axis labels
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel(y_label)

    # Add legend

    print(output_file)
    fig.savefig(output_file)
    plt.show()


def plot_metrics(args):
    pass


if __name__ == "__main__":
    pass


# location to save image
# data sources (allow multiple)
# x axis labels
# y axis label
# plot title
