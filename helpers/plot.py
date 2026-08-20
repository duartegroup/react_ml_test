"""Helpers for plotting."""
from typing import List, Literal, Union
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from react_ml_test.helpers.settings import DPI, SINGLE_COLUMN_FIG_SIZE, PlotSettings
from react_ml_test.helpers.utils import min_max_scale
from react_ml_test.helpers.logging import logger
from matplotlib.ticker import MaxNLocator


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def large_plot_2d(
    datasets: List[np.ndarray],
    colours: List[str],
    dataset_labels: List[str],
    plot_width: int = 800,
    plot_height: int = 600,
    out_fp: str = None,
    plot_settings: PlotSettings = PlotSettings()
):

    import datashader as ds
    import datashader.transfer_functions as tf

    # Input validation
    assert len(datasets) == len(colours) == len(dataset_labels), "datasets, colours, and dataset_labels must all have the same length."
    assert all([dataset.shape[1] == 2 for dataset in datasets]),  "all datasets must be 2D..."

    # create pandas dataframes based on 2D input np arrays
    data_frames = [pd.DataFrame({'x': dataset[:, 0].astype('float32'), 
                                 'y': dataset[:, 1].astype('float32'), 
                                 'label': dataset_label}) 
                   for dataset, dataset_label in zip(datasets, dataset_labels)]

    full_df = pd.concat(data_frames, ignore_index=True)
    full_df["label"] = full_df["label"].astype("category")

    # Create canvas
    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

    # Categorical aggregation
    agg = canvas.points(full_df, 'x', 'y', agg=ds.count_cat('label'))

    # Create a color key mapping label -> color
    colours = [mpl.colors.to_hex(colour) for colour in colours]
    color_key = dict(zip(dataset_labels, colours))

    # Shade using color_key
    img = tf.shade(agg, color_key=color_key, name="POINTS")
    img_pil = img.to_pil()

    # Display with matplotlib legend
    plt.figure(figsize=(plot_width / 100, plot_height / 100))
    plt.imshow(img_pil)

    # style plot
    # plt.xlabel(plot_settings.x_label)
    # plt.ylabel(plot_settings.y_label)
    # if plot_settings.xlim is not None:
    #     plt.xlim(plot_settings.xlim)
    # if plot_settings.ylim is not None:
    #     plt.ylim(plot_settings.ylim)

    if plot_settings.show_legend:
        patches = [Patch(color=col, label=lbl) for col, lbl in zip(colours, dataset_labels)]
        plt.legend(handles=patches, loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_fp)
    plt.show()


def plot_2d(datasets: List[np.array],
            save_fp: str,
            colour_maps: List[np.array] = None,
            title=None,
            dataset_names: List[str] = None,
            plot_settings: PlotSettings = PlotSettings(),
            ref_points: np.array = None,
            ref_point_labels: list[str] = None,
            show: bool = False
            ):
    """
    Plot a 2D scatter plot.

    :param datasets: a list of 2D datasets to plot each with dim (n_samples, 2)
    :param title: Title of the plot
    :param xlabel: Label for x-axis
    :param ylabel: Label for y-axis
    :param color: Optional color array for the points
    """

    cmap = mpl.cm.get_cmap('inferno')
    discrete_colours = plt.cm.tab10(np.arange(0, len(datasets) + 1) % 10)

    plt.figure(figsize=plot_settings.fig_size, dpi=plot_settings.dpi)

    # plot reference points and their labels, if specified
    if ref_points is not None:
        plt.scatter([x[0] for x in ref_points], [x[1] for x in ref_points], color='black', zorder=2, s=1.0, marker='x')
        if ref_point_labels is not None:
            for i, label in enumerate(ref_point_labels):
                plt.text(ref_points[i][0], ref_points[i][1] + 0.1, label, fontsize=6, ha='right')

    for i, dataset in enumerate(datasets):
        marker = plot_settings.custom_markers[i] if plot_settings.custom_markers is not None else None
        if colour_maps is not None:
            colour_map = normalize(colour_maps[i]) if plot_settings.normalise_colour_bar else colour_maps[i]
            plt.scatter(dataset[:, 0], dataset[:, 1], c=colour_map, marker=marker, alpha=0.8, cmap=cmap, s=0.8)
        else:
            colour = plot_settings.custom_colour_list[i] if plot_settings.custom_colour_list is not None else discrete_colours[i]
            plt.scatter(dataset[:, 0], dataset[:, 1], 
                        alpha=1.0, color=colour, marker=marker, s=2.0, 
                        label=dataset_names[i] if dataset_names else None)

    # format plots
    if plot_settings.title is not None:
        plt.title(title)
    if plot_settings.show_legend:
        plt.legend(fontsize=4, loc="upper left")
    plt.xlabel(plot_settings.x_label, fontsize=10)
    plt.ylabel(plot_settings.y_label, fontsize=10)
    if plot_settings.xlim is not None:
        plt.xlim(plot_settings.xlim)
    if plot_settings.ylim is not None:
        plt.ylim(plot_settings.ylim)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if plot_settings.x_label is None:
        plt.xticks([])
    if plot_settings.y_label is None:
        plt.yticks([])
    if colour_maps is not None:
        plt.colorbar(label=plot_settings.colour_bar_label)
    plt.tight_layout()
    plt.savefig(save_fp, format='png')
    logger.info(f"2D plot saved to '{save_fp}...'")
    if show:
        plt.show()

def heatmap_2D(data: np.ndarray, save_fp: str, cmap='coolwarm'):
    """
    Plots a 2D heatmap with a color bar mapped between -1.0 and 1.0.
    
    Parameters:
    data (numpy.ndarray): 2D array of values to be plotted as a heatmap.
    cmap (str): Colormap to be used. Default is 'coolwarm'.
    """
    plt.figure(figsize=(8, 6))
    import seaborn as sns
    ax = sns.heatmap(data, cmap=cmap, vmin=-1.0, vmax=1.0, cbar=True)
    plt.title("2D Heatmap with Color Bar (-1 to 1)")
    plt.savefig(save_fp, format='png')


def line_plot(x_datasets: List[np.ndarray],
              y_datasets: List[np.ndarray],
              dataset_labels: List[str],
              save_fp: str,
              y_errors: List[np.ndarray] = None,
              second_ax_points: List[tuple] = None,
              plot_settings: PlotSettings = PlotSettings()
              ):
    """
    Save a line plot of multiple x and y datasets.

    Parameters:
        x_datasets List[np.ndarray]: a list of datasets (to be matched with the provided y_datasets) representing a series on the line plot.
        y_datasets List[np.ndarray]: 
        second_ax_points List[tuple]: a list of points to be added to a broken axis portion of the plot (i.e. very high x values w.r.t rest of the data for comparison)

    """
    
    if plot_settings.custom_colour_list is None:
        colours = plot_settings.colour_scheme(np.arange(0, len(dataset_labels)) % 10)
    else:
        colours = plot_settings.custom_colour_list
    assert len(x_datasets) == len(y_datasets) == len(dataset_labels) == len(colours), f'x_datasets: {len(x_datasets)}, y_datasets: {len(y_datasets)}, dataset_labels: {len(dataset_labels)}, colours: {len(colours)}'

    plot_second_axis = False
    x_lim = (min(x_datasets[0]) - 2, max(x_datasets[1]))
    if second_ax_points is not None:
        fig, (ax, ax_2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': (10, 1)}, figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
        x_lim = (x_lim[0], x_lim[1] + 3)
        x2_lim = (min([x[0] for x in second_ax_points]) - 3, max([x[0] for x in second_ax_points]) + 3)
        plot_second_axis = True
    else:
        fig, ax = plt.subplots(1, 1, figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
        second_ax_points = [None for _ in range(len(x_datasets))]

    for i, (x_dataset, y_dataset, dataset_label, colour, ax_2_pt) in enumerate(zip(x_datasets, y_datasets, dataset_labels, colours, second_ax_points)):

        # check
        colour_dict_colour = plot_settings.colour_dict.get(dataset_label) if plot_settings.colour_dict is not None else None
        colour = colour_dict_colour if colour_dict_colour is not None else colour
        z_order = plot_settings.custom_zorder_list[i] if plot_settings.custom_zorder_list is not None else 1
        linestyle, z_order, alpha, point_linewidth = ('dotted', 2, 0.3, 0.2) if i in plot_settings.reference_indices else ('-', z_order, 0.9, 0.2)
        alpha = plot_settings.custom_alpha_list[i] if plot_settings.custom_alpha_list is not None else alpha

        # make correction to 0 value for x log scale
        if plot_settings.use_log_x_scale:
            x_dataset = x_dataset.copy()
            x_dataset[0] += 0.1

        ax.plot(x_dataset, y_dataset, label=dataset_label, linestyle=linestyle, color=colour, zorder=z_order, alpha=alpha, linewidth=2.0)

        # additional broken ax
        if ax_2_pt is not None:
            # ax_2.plot(ax_2_pt[0], ax_2_pt[1], color=colour, zorder=z_order, marker='o', markersize=1)
            ax_2.scatter(ax_2_pt[0], ax_2_pt[1], color=colour, zorder=z_order, linestyles=linestyle, s=8, alpha=alpha, linewidths=point_linewidth)

        # plot y-errors as shaded region if specified
        if y_errors is not None:
            ax.fill_between(x_dataset, y_dataset - y_errors[i], y_dataset + y_errors[i], color=colour, alpha=0.3, linewidth=0)

    if plot_settings.horizontal_ref_line is not None:
        ax.axhline(y=plot_settings.horizontal_ref_line,  color = 'black', linestyle='--')
    
    ax.set_xlabel(plot_settings.x_label, fontsize=10)
    ax.set_ylabel(plot_settings.y_label, fontsize=10)
    
    if plot_settings.use_log_y_scale:
        ax.set_yscale("log")
    
    if plot_settings.use_log_x_scale:
        ax.set_xscale("log")
    
    if plot_settings.xlim is not None:
        ax.set_xlim(plot_settings.xlim)

    if plot_settings.ylim is not None:
        ax.set_ylim(plot_settings.ylim)

    if plot_settings.reverse_x:
        ax.invert_xaxis()

    if plot_settings.reverse_y:
        ax.invert_yaxis()

    if plot_settings.title is not None:
        ax.set_title(plot_settings.title)

    if plot_settings.show_legend:

        if plot_settings.cust_legend_colours is not None and plot_settings.cust_legend_labels is not None and plot_settings.cust_legend_linestyles is not None:
            legend_handles = [Line2D([0], [0], color=col, label=lbl, linestyle=linestyle) 
                              for col, lbl, linestyle in zip(plot_settings.cust_legend_colours, plot_settings.cust_legend_labels, plot_settings.cust_legend_linestyles)]

            ax.legend(handles=legend_handles, fontsize=plot_settings.legend_fontsize, loc=plot_settings.legend_loc, frameon=False, ncols=len(x_datasets) / 2)
        else:
            ax.legend(fontsize=4, loc=plot_settings.legend_loc, frameon=False)
    
    if plot_settings.grid_lines:
        ax.grid(color='grey', alpha=0.1)

    if plot_settings.x_num_dp is not None:
        ax.xaxis.set_major_formatter(
            mticker.FormatStrFormatter(f'%.{plot_settings.x_num_dp}f')
        )

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    if plot_second_axis:

        ax.set_xticks(np.arange(0, x_lim[1], 10))

        # Set limits
        ax.set_xlim(*x_lim)
        ax_2.set_xlim(*x2_lim)

        # Exact tick for far axis
        ax_2.set_xticks([x[0] for x in second_ax_points])
        ax_2.set_xticklabels([str(x[0]) for x in second_ax_points], fontsize=10)
        
        # Hide spines
        ax.spines['right'].set_visible(False)
        ax_2.spines['left'].set_visible(False)

        # Remove duplicate y-axis on right
        ax_2.yaxis.set_visible(False)

        # Draw break marks
        d = 0.015
        kwargs_main = dict(transform=ax.transAxes, clip_on=False, color='black', linewidth=1)
        ax.plot((1-d, 1+d), (-d, +d), **kwargs_main)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs_main)

        kwargs_far = dict(transform=ax_2.transAxes, clip_on=False, color='black', linewidth=1)
        ax_2.plot((-d, +d), (-d, +d), **kwargs_far)
        ax_2.plot((-d, +d), (1-d, 1+d), **kwargs_far)

        plt.subplots_adjust(wspace=0.0)
    

    fig.tight_layout()
    fig.savefig(save_fp, format='png')

    print(f"Line plot saved as '{save_fp}...'")


def index_plot(datasets: List[np.array],
               dataset_labels: List[str] = [''],
               save_fp: str = None,
               plot_as_lines: List[int] = [],
               highlight_indices: np.ndarray = None,
               fig_size: tuple = SINGLE_COLUMN_FIG_SIZE,
               plot_peak_line: bool = False,
               shade_region: tuple = None,
               plot_settings: PlotSettings = PlotSettings(),
               ref_points: np.array = None,
               ref_point_labels: list[str] = None,
               ref_label_offsets: list[float] = [2.0]
):
    """
    Creates a scatter / line plot of the dataset against its index and saves it as a PNG file. The first dataset in
    the datasets list is assumed to be the reference dataset if multiple are specified.

    Parameters:
        data (iterable): The dataset to plot.
        save_fp (str): File path to save the output plot as a PNG.
        highlight_indices (list, optional): Indices of points to highlight with a different color.
        plot_as_lines (list, optional): list of indices of the input datasets to plot as lines.
    """
    plt.figure(figsize=fig_size, dpi=DPI)

    indices = np.arange(0, len(datasets[0]))

    # get discrete or continuous colour map
    if plot_settings.colour_map_type == 'discrete':
        colours = [colour for colour in plot_settings.colour_scheme(np.arange(0, len(datasets) + 1))]
    else:
        colours = plot_settings.colour_scheme(np.linspace(0.3, 1.0, len(datasets) + 1))

    # override with custom colour list
    colours = plot_settings.custom_colour_list if plot_settings.custom_colour_list is not None else colours
    alphas = plot_settings.custom_alpha_list if plot_settings.custom_alpha_list is not None else [1.0] * len(datasets)
     
    assert all([len(dataset) == len(indices) for dataset in datasets])

    colour_index, alpha_index = 0, 0
    for i, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):

        if i in plot_settings.reference_indices:
            colour = "black"
            alpha = 1.0
            z_order = 3
        else:
            colour = colours[colour_index]
            alpha = alphas[alpha_index]
            colour_index += 1
            alpha_index += 1
            z_order = 1
        
        line_style = plot_settings.custom_linestyle_list[i] if plot_settings.custom_linestyle_list is not None else '-'
        line_style = "--" if i in plot_settings.reference_indices else line_style
        line_width = 1.0 if i in plot_settings.reference_indices else plot_settings.line_width

        if i in plot_as_lines:
            plt.plot(indices, dataset, label=dataset_label, linestyle=line_style, 
                     c=colour, zorder=z_order, linewidth=line_width,
                     alpha=alpha)
        else:
            if highlight_indices is not None:
                highlight_mask = np.zeros(len(dataset), dtype=bool)
                highlight_mask[highlight_indices] = True

                normal_indices = indices[~highlight_mask]
                highlight_indices = indices[highlight_mask]

                plt.scatter(normal_indices, dataset[~highlight_mask])
                plt.scatter(highlight_indices, dataset[highlight_mask], color='orange', s=plot_settings.point_size)
            else:
                plt.scatter(indices, dataset, label=dataset_label, s=plot_settings.point_size, c=colour, zorder=z_order, alpha=alpha)

    # plot reference points and their labels, if specified
    if ref_points is not None:
        plt.scatter([x[0] for x in ref_points], [x[1] for x in ref_points], color='black', zorder=2, s=2.0)
        if ref_point_labels is not None:
            for i, label in enumerate(ref_point_labels):
                plt.text(ref_points[i][0], ref_points[i][1] + ref_label_offsets[i], label, fontsize=12, ha='center')

    if shade_region is not None:
        plt.axvspan(shade_region[0], shade_region[1], color='lightgrey', alpha=0.5, edgecolor=None, linewidth=0)

    # ensure only integers on x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel(plot_settings.x_label)
    plt.ylabel(plot_settings.y_label)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if plot_settings.use_log_y_scale:
        plt.yscale("log")
    if plot_settings.title is not None:
        plt.title(plot_settings.title)

    if plot_settings.show_legend:

        if plot_settings.cust_legend_colours is not None and plot_settings.cust_legend_labels is not None and plot_settings.cust_legend_linestyles is not None:
            legend_handles = [Line2D([0], [0], color=col, label=lbl, linestyle=linestyle) 
                              for col, lbl, linestyle in zip(plot_settings.cust_legend_colours, plot_settings.cust_legend_labels, plot_settings.cust_legend_linestyles)]

            plt.legend(handles=legend_handles, fontsize=plot_settings.legend_fontsize, loc=plot_settings.legend_loc, frameon=False, ncol=plot_settings.legend_ncols)
        else:
            plt.legend(fontsize=plot_settings.legend_fontsize, loc=plot_settings.legend_loc, frameon=False, ncol=plot_settings.legend_ncols)

    if plot_settings.highlight_region is not None:
        bl_x, bl_y, w, h = plot_settings.highlight_region
        plt.gca().add_patch(Rectangle((bl_x, bl_y), w, h, color=plot_settings.highlight_colour, alpha=0.15, edgecolor='none'))

    if plot_settings.grid_lines:
        plt.grid(color='grey', alpha=0.1)

    if plot_settings.ylim is not None:
        plt.ylim(plot_settings.ylim)

    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp, format='png', dpi=600)


def correlation_scatter_plot(ground_truth_set: np.ndarray, 
                             predicted_sets: List[np.ndarray],
                             prediction_names: List[str],
                             save_fp: str = None,
                             plot_settings: PlotSettings = PlotSettings(),
                             normalise: bool = True,
                             show: bool = False
                             ):
    """
    Creates a scatter plot to visualize the correlation between ground truth and predictions.

    Parameters:
        ground_truth (np.ndarray): Array of ground truth values.
        predicted List(np.ndarray): Array of predicted values.

    Returns:
        None
    """

    plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
    colours = plt.cm.tab10(np.arange(len(prediction_names)) % 10) if plot_settings.custom_colour_list is None else plot_settings.custom_colour_list

    # make all datasets relative to their minimum
    # plot predictions for each model against each respective ground truth set
    if normalise:
        norm_xs =  min_max_scale(ground_truth_set)
        norm_ys_sets = [np.copy(min_max_scale(ys)) for ys in predicted_sets]
        plt.plot(norm_xs, norm_xs, 'k-', linewidth=1, label='DFT')  # plot x=y line
    else:
        norm_xs = ground_truth_set
        norm_ys_sets = predicted_sets
        min_x = min(norm_xs)
        min_y = min([min(norm_ys) for norm_ys in norm_ys_sets])
        max_x = max(norm_xs)
        max_y = max([max(norm_ys) for norm_ys in norm_ys_sets])
        plt.plot((min_x, max_x), (min_y, max_y), 'k-', linewidth=1, label='DFT')  # plot x=y line

    # plot predictions for each model against each respective ground truth set
    for i, (norm_ys, prediction_name) in enumerate(zip(norm_ys_sets, prediction_names)):
        
            # validate inputs
            if norm_xs.shape != norm_ys.shape:
                raise ValueError(f"Ground_truth and predictions must have the same shape but have shapes: {norm_xs.shape} and {norm_ys.shape}...")

            # plot points
            plt.scatter(norm_xs, 
                        norm_ys, 
                        alpha=plot_settings.custom_alpha_list[i] if plot_settings.custom_alpha_list is not None else 0.8, 
                        edgecolor='k', 
                        linewidths=plot_settings.cust_marker_linewidth[i] if plot_settings.cust_marker_linewidth is not None else 0.2,  
                        s=plot_settings.cust_size_list[i] if plot_settings.cust_size_list is not None else 4, 
                        label=plot_settings.cust_legend_labels[i] if plot_settings.cust_legend_labels is not None else prediction_name,
                        color=colours[i],
                        marker=plot_settings.custom_markers[i] if plot_settings.custom_markers is not None else 'o')

    # add labels and title
    plt.xlabel(plot_settings.x_label)
    plt.ylabel(plot_settings.y_label)
    if plot_settings.show_legend:
        plt.legend(fontsize=4, loc="upper left", frameon=False)
    plt.title(plot_settings.title)
    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp, format='png')
    if show:
        plt.show()
    plt.close()
    logger.info(f"Correlation plot saved as {save_fp}...")


def histogram_1D(ds_list: List[np.ndarray],
                 output_file: str,
                 bins: int = 100,
                 title: str = None,
                 x_label: str = 'Value',
                 y_label: str = 'Frequency',
                 labels: List[str] = None):
    """
    Creates and saves a histogram from a numpy array.

    Parameters:
        data (numpy.ndarray): The input data for the histogram.
        output_file (str): The name of the output PNG file. Default is 'histogram.png'.
        bins (int): Number of bins in the histogram. Default is 10.
    """

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each dataset
    for i, data in enumerate(ds_list):
        label = labels[i] if labels else None
        ax.hist(data, bins=bins, alpha=0.6, label=label, linewidth=0.5)

    # Add titles and labels
    if title:
        ax.set_title(title, fontsize=16, weight='bold')

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Add legend
    ax.legend(fontsize=8)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Histogram saved as '{output_file}...'")


def bar_plot(datasets: List[np.array],
             dataset_labels: List[str],
             x_data: Union[List[str], List[int]],
             save_fp: str,
             mode: Literal['category', 'numerical'] = 'category',
             plot_settings: PlotSettings = PlotSettings(),
             show: bool = False,
             show_x_tick_labels: bool = True,
             ):

    assert mode in {'category', 'numerical'}, 'mode must be one of: [category, numerical]'

    plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)

    x = np.arange(len(datasets[0]))
    width = 1.0 / float(len(x_data) * 1.2)

    colours = plot_settings.colour_scheme(np.arange(len(x_data)))
    colours = plot_settings.custom_colour_list if plot_settings.custom_colour_list is not None else colours
    alphas = plot_settings.custom_alpha_list if plot_settings.custom_alpha_list is not None else [1.0 for _ in range(len(colours))]

    if mode == 'category':

        for i, (category, colour, alpha) in enumerate(zip(x_data, colours, alphas)):
            if any(x is None for x in datasets[i]):
                continue  # skip if null value encountered
            plt.bar(x + i*width, datasets[i], width, label=category, color=colour, alpha=alpha)
       
        if show_x_tick_labels:
            # plt.xticks(x + width * 1, dataset_labels, rotation=30, ha="right", fontsize=10)
            plt.xticks(x + width * 1, dataset_labels, fontsize=10)
        else:
            plt.xticks(x + width * 1, list(range(len(dataset_labels))), fontsize=10)
    else:
        for dataset in datasets:
            if any(x is None for x in dataset):
                continue  # skip if null value encountered
            plt.bar(x_data, dataset)

    # Formatting
    if plot_settings.use_log_y_scale:
        plt.yscale("log")

    if plot_settings.integers_for_y:
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    if plot_settings.ylim is not None:
        plt.ylim(plot_settings.ylim)

    plt.gca().set_position([0.18, 0.15, 0.75, 0.72])

    plt.xlabel(plot_settings.x_label, fontsize=10)
    plt.ylabel(plot_settings.y_label, fontsize=10)
    plt.yticks(fontsize=10)

    if plot_settings.title is not None:
        plt.title(plot_settings.title)
    if plot_settings.show_legend:
        plt.legend(fontsize=4, loc="upper left")
    
    # plt.tight_layout()
    plt.savefig(save_fp)
    print(f"Saved bar plot to {save_fp}...")
    if show:
        plt.show()
    plt.close()

def violin_plot(model_datasets: List[List[np.array]],
                model_names: List[str],
                dataset_names: List[str],
                save_fp: str = None,
                plot_settings: PlotSettings = PlotSettings(),
               ):

    fig, ax = plt.subplots(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
    assert plot_settings.num_base_models is not None, "For a violin plot you need to specify the number of base models to compare."
    assert plot_settings.num_base_model_comps is not None, "For a violin plot you need to specify the number of comparisons per base model to make."
    # assert len(dataset_names) == len(model_datasets[0])
    
    n = plot_settings.num_base_models
    possible_offsets = [(k - (n - 1) / 2) * 0.16 for k in range(n)]
    clip_offsets = [off for k in range(n) for off in [1, 0]]
    x_offsets = [off for off in possible_offsets for _ in range(plot_settings.num_base_model_comps)]
    for j, model_name in enumerate(model_names):
        positions = [k + 1 + x_offsets[j] for k in range(len(dataset_names))]
        y = [model_datasets[i][j] for i in range(len(dataset_names))]
        parts = ax.violinplot(y, positions=positions, widths=0.22, showmeans=False, showmedians=True)

        for k, body in enumerate(parts["bodies"]):
            body.set_facecolor(plot_settings.custom_colour_list[j])
            body.set_edgecolor("black")
            body.set_alpha(plot_settings.custom_alpha_list[j])
            clip = Rectangle((positions[k] - clip_offsets[j], -1e6), 1, 2e6, transform=ax.transData)
            body.set_clip_path(clip)

        for comp in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[comp].set_edgecolor("black")
            parts[comp].set_linewidth(0.8)
            parts[comp].set_alpha(plot_settings.custom_alpha_list[j])

    ax.set_xticks(list(range(1, len(dataset_names) + 1)))
    ax.set_xticklabels(dataset_names, fontsize=10)
    ax.set_xlim(0.5, len(dataset_names) + 0.5)
    ax.tick_params(axis='y', labelsize=10)
    if plot_settings.ylim is not None:
        ax.set_ylim(plot_settings.ylim)
    if plot_settings.use_log_y_scale:
        plt.yscale("log")
    # else:
    #     plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_xlabel(r"Config Type", fontsize=10)
    ax.set_ylabel(plot_settings.y_label, fontsize=10)

    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp, dpi=DPI)
        logger.info(f"Saved violin plot to {save_fp}...")
    plt.close()
