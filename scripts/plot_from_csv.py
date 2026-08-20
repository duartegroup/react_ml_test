"""
Script to plot a correlation plot from two random input csv headers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append('../')
from react_ml_test.helpers.plot import correlation_scatter_plot, line_plot, bar_plot
from react_ml_test.helpers.settings import CSV_DIR, PLOT_DIR, GUAJADIAL_DA_DATA_DIR, PlotSettings, PAPER_MODEL_COLOURS, OUT_DIR
from react_ml_test.helpers.io import get_name_from_path


def plot_pairwise_correlation(csv_fname: str, 
                              x_column_header: str, 
                              y_column_header: str,
                              save_fp: str):
    import seaborn as sns

    # Load the data from the CSV file
    df = pd.read_csv(csv_fname)
    
    # Check if the specified columns exist in the dataframe
    if x_column_header not in df.columns or y_column_header not in df.columns:
        print(f"Error: Columns '{x_column_header}' and/or '{y_column_header}' not found in the CSV file.")
        return
    
    # Create a new DataFrame with the x and y values
    data = df[[x_column_header, y_column_header]]
    
    # Create a pairwise correlation plot using seaborn's pairplot
    sns.pairplot(data)
    
    # Add titles and labels
    plt.title(f"Pairwise Correlation Plot: {x_column_header} vs {y_column_header}")
    plt.savefig(save_fp, format='png')


def plot_category_scatter_with_error_bars(csv_fpath: str, 
                                          category_column_header: str, 
                                          y_column_header: str, 
                                          error_column_header: str,
                                          save_fp: str):
    import seaborn as sns
    
    # Load the data from the CSV file
    df = pd.read_csv(csv_fpath)
    
    # Check if the specified columns exist in the dataframe
    required_columns = [y_column_header, category_column_header, error_column_header]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in the CSV file.")
            return
    
    # Create a scatter plot with categories on the x-axis
    plt.figure(figsize=(8, 6))
    
    # Create a scatter plot, where the categories are on the x-axis and the values are on the y-axis
    sns.barplot(data=df, x=category_column_header, y=y_column_header, ci=None, color='b', capsize=0.1)
        
    # Add error bars
    plt.errorbar(df[category_column_header], df[y_column_header], yerr=df[error_column_header], fmt='none', ecolor='gray', capsize=3)
    
    # Add titles and labels
    plt.title(f"Category Scatter Plot: {category_column_header} vs {y_column_header} (with error bars)")
    plt.xlabel(category_column_header)
    plt.ylabel(y_column_header)

    print(f"Saving {save_fp}...")
    plt.savefig(save_fp, format='png')


if __name__ == "__main__":

    plot_settings = PlotSettings()
    plot_type = None

    # ===== ENTER csv_fpath, x_column_name and y_column_name, and optionally, error bars here =====

    # --- timings bar plots ---
    # df = pd.read_csv(f"{OUT_DIR}/timing/combined_concatenated_md_trajectory_times.csv")
    # means = df.groupby("Model Name", sort=False)[["Time per Traj", "Time per Step"]].mean().reset_index()
    # stds = df.groupby("Model Name", sort=False)[["Time per Traj", "Time per Step"]].std().reset_index()

    # plot_settings = PlotSettings(x_label="Model Name", 
    #                              y_label="Time per Step (s)", 
    #                              custom_colour_list=list(PAPER_MODEL_COLOURS.values(),
    #                              ))
    # save_fp = f"{OUT_DIR}/timing/mean_time_per_traj.png"
    # bar_plot([means["Time per Traj"].to_numpy()], means["Model Name"].tolist(), ["Mean Time per Traj"],
    #         save_fp=save_fp, plot_settings=plot_settings)
    # print(f'Saved {save_fp}')

    # plot_settings = plot_settings._replace(x_label="Model Name", y_label="Time per Step (s)")
    # save_fp = f"{OUT_DIR}/timing/mean_time_per_step.png"
    # print(f'Saved {save_fp}')

    # bar_plot([means["Time per Step"].to_numpy()], 
    #           means["Model Name"].tolist(), 
    #           ["Mean Time per Step"], save_fp=save_fp, plot_settings=plot_settings)
    # print("=== MEANS ===")
    # print(means)
    # print("=== STDS ===")
    # print(stds)

    # --- category plot ---
    # csv_fpath = f'{CSV_DIR}/foundation_models_soap_desc_analysis.csv'  # Replace with your CSV file path
    # category_column_header = 'Test Dataset'  # Replace with your x column header
    # y_column_header = 'Cos. Sim. Mean'  # Replace with your y column header
    # error_column_header = 'Cos. Sim. SD'  # Replace with the column header for error values
    # save_fp = f"{PLOT_DIR}/descriptors/{get_name_from_path(csv_fpath)}_category_plot.png"

    # plot_category_scatter_with_error_bars(csv_fpath, 
    #                                       category_column_header, 
    #                                       y_column_header, 
    #                                       error_column_header, 
    #                                       save_fp)
    
    # correlation plot for soap desc analysis
    csv_fpath = f'{CSV_DIR}/foundation_models_soap_desc_analysis.csv'  # Replace with your CSV file path
    x_column_header = 'Cos. Sim. Mean'  # Replace with your y column header
    y_column_header = ''  # Replace with your y column header
    save_fp = f"{PLOT_DIR}/descriptors/{get_name_from_path(csv_fpath)}_correlation_plot.png"
    plot_type = 'corr'

    # --- fes line_plot ---
    save_name = 'guajadial_DA_OFF50_FES'
    runs = [100, 200, 300, 400, 450]
    dataset_labels = [f"{i} ps" for i in runs]
    baseline_values = [-19.9, -20.0, -20.1, -20.15, -20.16]
    relative_ref_index = -1
    # csv_fpaths = [f'{GUAJADIAL_DA_DATA_DIR}/fes/{save_name}_{i}ps.csv' for i in runs]  # each CSV fpath contains a series
    csv_fpaths = [f'{GUAJADIAL_DA_DATA_DIR}/fes/{save_name}_{i}ps_filtered.csv' for i in runs]  # each CSV fpath contains a series
    x_column_header = 'Coord'  # Replace with your y column header
    y_column_header = 'Free'  # Replace with your y column header
    save_fp = f"{PLOT_DIR}/line_plots/{save_name}_line_plot.png"
    plot_type = 'line'
    plot_settings = plot_settings._replace(custom_colour_list = [PAPER_MODEL_COLOURS['MACE-OFF23_medium']] * len(dataset_labels), 
                                           custom_alpha_list=[0.2 * i for i in range(1, len(dataset_labels) + 1)], # (len(model_names) // 2),
                                           show_legend=False,
                                           horizontal_ref_line = 0.0,
                                           x_label = '(r1 + r2)/2 distance (A)',
                                           y_label = 'PMF (kcal / mol)',
                                           reverse_x = True,
                                           xlim = (1.4, 4.5),
                                           x_num_dp = 1,
                                           legend_fontsize = 10,
                                           #  cust_legend_labels = [f"{get_formatted_model_name(model_name)}" for model_name in PAPER_MODELS] + [f"{get_formatted_model_name(model_name)} TS FT"  for model_name in PAPER_MODELS],
                                          )
    
    # Load the data from the CSV file to get x and y series
    dfs = [pd.read_csv(csv_fpath) for csv_fpath in csv_fpaths]
    
    # Check if the specified columns exist in the dataframe
    x_series, y_series = [], []
    for i, df in enumerate(dfs):
        if x_column_header not in df.columns or y_column_header not in df.columns:
            print(f"Error: Columns '{x_column_header}' and/or '{y_column_header}' not found in the CSV file.")
            sys.exit(-1)
    
        # update new DataFrame with the x and y values
        # df = df[[x_column_header, y_column_header]]
        x = np.array(df[x_column_header])
        y =  np.array(df[y_column_header])
        ref_val = y[relative_ref_index]
        y = y - ref_val
        # y = baseline_values[i] + np.array(df[y_column_header])
        x_series.append(x)
        y_series.append(y)
    
    print('Coords: ', x_series[-1])
    print('Delta Gs: ', y_series[-1])
    print('Max Delta G: ', np.max(y_series[-1]))

    if plot_type == 'corr':
        correlation_scatter_plot(csv_fpath, 
                                 x_column_header, 
                                 y_column_header,
                                 save_fp)  # TODO: wrong!
    
    elif plot_type == 'line':
        line_plot(x_series, y_series, dataset_labels, save_fp, plot_settings=plot_settings)

    # elif plot_type == 'bar':
    #     bar_plot(x_series, y_series, dataset_labels, save_fp, plot_settings=plot_settings)