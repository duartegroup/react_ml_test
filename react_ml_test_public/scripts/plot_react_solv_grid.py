
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    PowerNorm,
    TwoSlopeNorm,
)

from react_ml_test_public.helpers.settings import CSV_DIR, PLOT_DIR
from react_ml_test_public.helpers.utils import format_value_with_uncertainty

if __name__ == "__main__":

    # SELECT MODELS:
    model_names = ["GO-MACE-23", "MACE-OFF23_medium", "MACE-MP-0b3_medium"]

    # SELECT GRID SAMPLES RUN
    n = 8
    # n = 16

    # SOME OTHER SETTINGS
    set_relative_to_baseline = True
    set_min_to_zero = True
    sig_figs = 3
    show_grid_text: bool = True

    all_model_energy_errors, all_model_force_errors = [], []
    for model_name in model_names:

        # SELECT INPUT CSV FILE:

        # 1. --- CPD + MVK DA in water ---
        # input_csv_file = f"{CSV_DIR}/test_set_comparison_CPD_MVK_DA_test_{model_name}_endo_DA_grid_n={n}_fine_tuned_endo_DA_n={8}_r_dist.csv"
        # # save_name, col_name = 'CPD_MVK_test', 'All'
        # # save_name, col_name = 'CPD_MVK_test_vac_only', 'Vacuum'
        # save_name, col_name = 'CPD_MVK_test_solv_only', 'Solvated'
        # # save_name, col_name = 'CPD_MVK_test_solv_only_reactants', 'Solvated Reactants'
        # # save_name, col_name = 'CPD_MVK_test_solv_only_solvent', 'Solvent'

        # 2. --- anthracene DA in toluene ---
        input_csv_file = f"{CSV_DIR}/test_set_comparison_anthracene_DA_test_{model_name}_anthracene_DA_grid_n={n}_fine_tuned_anthracene_DA_n={n}_r_dist.csv"
        # # save_name, col_name = 'anthracene_DA_test', 'All'
        # # save_name, col_name = 'anthracene_DA_test_vac_only', 'Vacuum'
        save_name, col_name = 'anthracene_DA_test_solv_only', 'All Solvated'
        # save_name, col_name = 'anthracene_DA_test_solv_only_reactants', 'Solvated Reactants'
        # save_name, col_name = 'anthracene_DA_test_solv_only_solvent', 'Solvent'

        # 3. --- guajadial ---
        # input_csv_file = f"{CSV_DIR}/test_set_comparison_guajadial_DA_test_{model_name}_guajadial_DA_grid_n={n}_fine_tuned_guajadial_DA_n={n}_r_dist.csv"
        # # save_name, col_name = 'guajadial_DA_test', 'All'
        # # save_name, col_name = 'guajadial_DA_test_vac_only', 'Vacuum'
        # save_name, col_name = 'guajadial_DA_test_solv_only', 'Solvated'
        # # save_name, col_name = 'guajadial_DA_test_solv_only_reactants', 'Solvated Reactants'
        # # save_name, col_name = 'guajadial_DA_test_solv_only_solvent', 'Solvent'

        # ===========================

        df = pd.read_csv(input_csv_file)

        # get base model 
        base_model_energy_error, base_model_force_error = df.iloc[0][f"{col_name} Energies (meV / atom)"], df.iloc[0][f"{col_name} Forces (meV / A)"]
        df = df.iloc[1:]  # remove base model row
        print("Base Model: ", model_name)
        print("Base Model Energy Error: ", base_model_energy_error)
        print("Base Model Force Error: ", base_model_force_error)

        # group by solv frac
        df.sort_values(['Model'])
        df['run'] = df['Model'].str.extract(r'^(.*?)(?:_rep_\d+|_fine_tuned)')[0]
        df['solv'] = df['Model'].str.extract(r'solv_(\d+)').astype(str)
    
        mean_df = df.copy(deep=True)
        std_df = df.copy(deep=True)

        mean_grouped = mean_df.groupby('run')
        std_grouped = std_df.groupby('run')

        mean_solv_values = mean_grouped['solv'].first()
        std_solv_values = std_grouped['solv'].first()

        mean_df = mean_grouped.mean(numeric_only=True)
        std_df = std_grouped.std(numeric_only=True)

        mean_df['solv'] = mean_solv_values
        std_df['solv'] = std_solv_values

        # grouped = df.groupby('solv')
        mean_grouped = mean_df.groupby('solv')
        energy_grid_rows, force_grid_rows = [], []
        for solv, group in mean_grouped:
            energy_grid_rows.append(np.array(group[f"{col_name} Energies (meV / atom)"]))
            force_grid_rows.append(np.array(group[f"{col_name} Forces (meV / A)"]))
        abs_energy_grid_rows, abs_force_grid_rows = np.array(energy_grid_rows), np.array(force_grid_rows)

        # print('Energy Grid Rows: \n', energy_grid_rows)
        # print('Force Grid Rows: \n', force_grid_rows)

        # get stdevs
        std_grouped = std_df.groupby('solv')
        energy_stds, force_stds = [], []
        for solv, group in std_grouped:
            energy_stds.append(np.array(group[f"{col_name} Energies (meV / atom)"]))
            force_stds.append(np.array(group[f"{col_name} Forces (meV / A)"]))

        energy_stds, force_stds = np.array(energy_stds), np.array(force_stds)
        # print('Energy Stds: \n', energy_stds)
        # print('Force Stds: \n', force_stds)

        if set_relative_to_baseline:
            energy_grid_rows = abs_energy_grid_rows - base_model_energy_error
            force_grid_rows = abs_force_grid_rows - base_model_force_error
            # print('\u0394 Energy Grid Rows: \n', energy_grid_rows)
            # print('\u0394 Force Grid Rows: \n', force_grid_rows)#
        else:
            energy_grid_rows = abs_energy_grid_rows
            force_grid_rows = abs_force_grid_rows

        use_log_scale = False
        if set_relative_to_baseline:
            label_set = ['\u0394 Energy (FT - Baseline) RMSE (meV / atom)', '\u0394 ' + r'Force (FT - Baseline) RMSE (meV / $\AA$)']
        else:
            label_set = ['Energy RMSE (meV / atom)', r'Force Comp. RMSE (meV / $\AA$)']
        type_set = ['Energy', 'Force']
        # y_ranges = [(-15, 15), (-730.0, 730.0)]
        y_ranges = [(None, None), (None, None)]
        abs_grid_rows_set = [abs_energy_grid_rows, abs_force_grid_rows]
        std_grid_rows_set = [energy_stds, force_stds]
        grid_rows_set = [energy_grid_rows, force_grid_rows]
        baseline_errors = [base_model_energy_error, base_model_force_error]
        for grid_rows, abs_grid_rows, std_grid_rows, baseline_error, label, out_type, y_range in zip(grid_rows_set, abs_grid_rows_set, std_grid_rows_set, baseline_errors, label_set, type_set, y_ranges):

            # normalise the data between -1.0 and 1.0
            # min_val = np.min(grid_rows)
            # max_val = np.max(grid_rows)
            # grid_rows = 2 * (grid_rows - min_val) / (max_val - min_val) - 1.0
            # print('Grid Rows Norm: \n', grid_rows)

            if use_log_scale:
                grid_rows = np.log(grid_rows)

            # get max point
            if set_relative_to_baseline:
                tick_range = np.max([abs(np.min(grid_rows)), abs(np.max(grid_rows))])  # if positve + negative
                min_tick = -tick_range

                # if set_min_to_zero:
                #     min_tick = 0
                #     tick_range = np.max([abs(np.min(grid_rows)), abs(np.max(grid_rows))])  # if set baseline to 0
                #     min_tick = 0
            else:
                min_tick = np.min(grid_rows)
                max_tick = 2 * baseline_error - np.min(grid_rows)

            # Create the plot
            fig, ax = plt.subplots()
            norm = None
            # norm = PowerNorm(gamma=0.6, vmin=y_range[0], vmax=y_range[1])
            # norm = TwoSlopeNorm(vmin=np.min(grid_rows), vmax=max(np.max(grid_rows), 0.1), vcenter=0)

            # Original colormap
            cmap = plt.get_cmap('RdYlBu_r')

            # Create truncated colormap (first half: 0.0 → 0.5)
            new_cmap = LinearSegmentedColormap.from_list(
                'RdYlBu_r_half',
                cmap(np.linspace(0.0, 0.5, 256))
            )

            if set_min_to_zero:
                norm = Normalize(vmin=np.min(grid_rows), vmax=0)
                im = ax.imshow(grid_rows, cmap=new_cmap, aspect='equal', norm=norm)
            else:
                norm = TwoSlopeNorm(vmin=min_tick, vmax=tick_range, vcenter=0) if set_relative_to_baseline else TwoSlopeNorm(vmin=None, vmax=None, vcenter=baseline_error)
                im = ax.imshow(grid_rows, cmap='RdYlBu_r', aspect='equal', norm=norm)
            # norm = TwoSlopeNorm(vmin=None, vmax=None, vcenter=baseline_error)
            # im = ax.imshow(grid_rows, cmap='viridis_r', aspect='equal')
            # im = ax.imshow(grid_rows, cmap='RdYlGn_r', aspect='equal', norm=norm)

            # print('Grid Rows: \n', grid_rows)
            # print('Abs Grid Rows: \n', abs_grid_rows)

            # Annotate each cell with its value
            if show_grid_text:
                grid_rows_max = np.max(np.abs(grid_rows))
                txt_colour_thresh = 0.80 * grid_rows_max

                for i in range(abs_grid_rows.shape[0]):
                    for j in range(abs_grid_rows.shape[1]):
                        formatted_val, formatted_std = format_value_with_uncertainty(abs_grid_rows[i, j], std_grid_rows[i, j])

                        # Create the label with value ± std
                        grid_cell_label = f"{formatted_val}\n± {formatted_std}"
                    
                        # Determine text color
                        if set_relative_to_baseline:
                            text_color = 'white' if abs(grid_rows[i, j]) > txt_colour_thresh else 'black'
                        else:
                            text_color = 'white' if abs_grid_rows[i, j] < txt_colour_thresh else 'black'
                    
                        ax.text(j, i, grid_cell_label, ha='center', va='center', color=text_color, fontsize=14)

            # Add a colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(label, rotation=270, labelpad=25, fontsize=12)  # Customize label and padding
            cbar.ax.tick_params(labelsize=14)
            # cbar.set_ticks()

            # set custom ticks
            # min_tick = np.min(grid_rows)
            # max_tick = 2 * baseline_error - np.min(grid_rows)
            # step = max_tick / 10.0
            # cbar.set_ticks(list(np.arange(min_tick, max_tick, step)) + [baseline_error])
            # cbar.set_ticklabels([baseline_error])

            # Optional: add grid lines
            ax.set_xticks(np.arange(grid_rows.shape[1]+1) - 0.5, minor=True)
            ax.set_yticks(np.arange(grid_rows.shape[0]+1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Optional: remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{model_name} {out_type}')
            label_str = '_labelled' if show_grid_text else ''
            save_fp = f"{PLOT_DIR}/grid/{model_name}_{save_name}_{out_type}_n={n}_react_solv_grid{label_str}.png"
            plt.savefig(save_fp, format='png', dpi=600)

            print(f'Saved plot to {save_fp}')
