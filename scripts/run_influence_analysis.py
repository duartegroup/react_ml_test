"""
Script to analyse the influence of data points on fine-tuning performance for a given pretrained model. 
Assumes you have run_test_set_comparison.py to generate npz files. 
"""

import csv
import glob
import os
import re
import sys
from time import time
from typing import Union

import ase
import chemiscope
import numpy as np

from react_ml_test.helpers.io import get_ase_atoms_energy
from react_ml_test.helpers.sampling import sample_dataset

sys.path.append('../')
import matplotlib.pyplot as plt

from react_ml_test.helpers.descriptors import get_mace_descriptors, get_soap_descriptors
from react_ml_test.helpers.dim_red import calc_best_similarities, run_pca
from react_ml_test.helpers.io import (
    autode_props_parser,
    flexible_ase_read,
    get_name_from_path,
    load_data_cache,
    load_hdf5_cache,
    save_npz_data_cache,
)
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.model_eval import (
    get_model_predictions,
    get_test_and_pred_data,
)
from react_ml_test.helpers.plot import bar_plot, correlation_scatter_plot, plot_2d
from react_ml_test.helpers.settings import (
    CSV_DIR,
    DATASETS_DIR,
    ENDO_DA_DATA_DIR,
    GRAPH_PES_MODEL_LOADER,
    ISOLATED_ATOMS_DIR,
    MACE_MODEL_LOADER,
    MODEL_DIR,
    MODELS,
    NPZ_DIR,
    PLOT_DIR,
    SAMPLES_OUT_DIR,
    TRAIN_DATA_DIR,
    TRAIN_SUBSETS_DIR,
    PlotSettings,
)
from react_ml_test.helpers.stats import calculate_comp_rmse, calculate_rmse
from react_ml_test.helpers.train_utils import WeightedEnergyForceMSELoss
from react_ml_test.helpers.utils import (
    get_ase_props,
    load_model_calculator,
    natural_sort_key,
    unflatten_array_by_lengths,
)

N_CORES = 1

def influence_analysis(test_configs: list[ase.Atoms],
                       test_energies: np.ndarray,
                       test_forces: Union[list, np.ndarray],
                       pred_energies_dict: dict[str, np.ndarray],
                       pred_forces_dict: dict[str, np.ndarray],
                       base_pred_energies: np.ndarray, 
                       base_pred_forces: np.ndarray,                       
                       model_names: list[str],
                       train_configs: list[ase.Atoms],
                       train_ds_name: str,
                       test_set_name: str,
                       save_name: str,
                       config_mean_best_soap_sims: list[float],
                       config_mean_best_mace_sims: list[float],
                       train_soap_desc_2D: list[np.ndarray],
                       train_config_pca_mace_descs: list[np.ndarray],
                       train_ts_distances: list[float],
                       r1_dists: list[float],
                       r2_dists: list[float],
                       ref_ts_point: tuple[float],
                       train_true_energy_deltas: list[float], 
                       train_pred_energy_deltas: list[float],
                       proponents_only: bool = False
                       ):
    """
    A disgusting function to run the analysis, that is in desperate need of a refactor.
    """
        
    # get base model energy RMSEs
    base_model_energy_rmse = calculate_rmse(base_pred_energies, test_energies)

    # get base model force RMSEs and losses 
    if test_forces is not None:
        base_model_force_rmse = calculate_comp_rmse(base_pred_forces, test_forces)
        base_model_loss = loss_func(test_configs, test_energies, base_pred_energies, test_forces, base_pred_forces, 
                                    energy_weight=energy_weight, forces_weight=forces_weight)
    
    # for each model representing a configuration in the dataset
    delta_energy_errors, delta_force_errors, delta_losses = [], [], []  # [(config_id, delta_value)]
    for model_name in model_names:
        
        match = re.search(r'config_(\d+)', model_name)
        config_index = int(match.group(1))

        # energies, get improvement over base model
        model_pred_energies = pred_energies_dict[model_name]          
        model_energy_rmse = calculate_rmse(model_pred_energies, test_energies)
        delta_energy_rmse = model_energy_rmse - base_model_energy_rmse
        delta_energy_errors.append((config_index, delta_energy_rmse))

        # forces
        if test_forces is not None and len(test_forces) > 0:
            model_pred_forces = pred_forces_dict[model_name]
            model_force_rmse = calculate_comp_rmse(model_pred_forces, test_forces)
            delta_force_rmse = model_force_rmse - base_model_force_rmse
            delta_force_errors.append((config_index, delta_force_rmse))

            # add loss
            model_loss = loss_func(test_configs, test_energies, model_pred_energies, test_forces, model_pred_forces,  
                                    energy_weight=energy_weight, forces_weight=forces_weight)
            delta_loss = model_loss - base_model_loss
            delta_losses.append((config_index, delta_loss))

    # do rankings
    ranked_delta_energy_errors = sorted(delta_energy_errors, key=lambda x: x[1])
    ranked_delta_force_errors = sorted(delta_force_errors, key=lambda x: x[1])
    ranked_delta_losses = sorted(delta_losses, key=lambda x: x[1])

    # save ds ordered by influence on test set
    loss_influence_sorted_configs = [train_configs[x[0]].copy() for x in ranked_delta_losses]
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_loss_influence_sorted.xyz", loss_influence_sorted_configs, format='extxyz')   

    logger.info(f'Top 10 influence energy configs: {ranked_delta_energy_errors[0:10]}')
    logger.info(f'Top 10 influence force configs: {ranked_delta_force_errors[0:10]}')
    logger.info(f'Top 10 influence loss configs: {ranked_delta_losses[0:10]}')
    logger.info(f'Bottom 10 influence energy configs: {ranked_delta_energy_errors[-10:]}')
    logger.info(f'Bottom 10 influence force configs: {ranked_delta_force_errors[-10:]}')
    logger.info(f'Bottom 10 influence loss configs: {ranked_delta_losses[-10:]}')

    # save most positively and negatively influential samples
    top_10_energy_configs = sample_dataset(train_ds_fp, specific_indices=[x[0] for x in ranked_delta_energy_errors[0:10]])
    top_10_force_configs = sample_dataset(train_ds_fp, specific_indices=[x[0] for x in ranked_delta_force_errors[0:10]])
    top_10_loss_configs = sample_dataset(train_ds_fp, specific_indices=[x[0] for x in ranked_delta_losses[0:10]])
    bottom_10_energy_configs = sample_dataset(train_ds_fp, specific_indices=[x[0] for x in ranked_delta_energy_errors[-10:]])
    bottom_10_force_configs = sample_dataset(train_ds_fp, specific_indices=[x[0] for x in ranked_delta_force_errors[-10:]])
    bottom_10_loss_configs = sample_dataset(train_ds_fp, specific_indices=[x[0] for x in ranked_delta_losses[-10:]])
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_top_10_energy_influence.xyz", top_10_energy_configs, format='extxyz')
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_top_10_force_influence.xyz", top_10_force_configs, format='extxyz')
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_top_10_loss_influence.xyz", top_10_loss_configs, format='extxyz')
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_bottom_10_energy_influence.xyz", bottom_10_energy_configs, format='extxyz')
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_bottom_10_force_influence.xyz", bottom_10_force_configs, format='extxyz')
    ase.io.write(f"{SAMPLES_OUT_DIR}/{train_ds_name}_{base_model_name}_{test_set_name}_bottom_10_loss_influence.xyz", bottom_10_loss_configs, format='extxyz')

    # optionally filter only proponents (i.e. with < 0 influence)  # TODO: this changes the shapes of our arrays... we need to account for this!
    if proponents_only:
        delta_energy_errors = [x for x in delta_energy_errors if x[1] < 0.0]
        delta_force_errors = [x for x in delta_force_errors if x[1] < 0.0]
        delta_losses = [x for x in delta_losses if x[1] < 0.0]

    delta_energy_error_values = np.array([x[1] for x in delta_energy_errors])
    delta_force_error_values = np.array([x[1] for x in delta_force_errors])
    delta_loss_values = np.array([x[1] for x in delta_losses])

    # loss delta bar plot
    if test_forces is not None and len(test_forces) > 0:
        plot_settings = PlotSettings(x_label='Sample', y_label='\u0394 Loss', use_log_y_scale=False, show_legend=False)
        bar_plot([delta_loss_values],
                ['Base Model Deltas'],
                range(len(delta_loss_values)),
                save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_set_name}_loss_influence_bar_plot.png",
                mode='numerical',
                plot_settings=plot_settings
                )

    # now correlate influence to data features
    y_labels = ['\u2206 Energy RMSE (meV / atom)', '\u2206 Force Comp. RMSE (meV / A)', '\u2206 Loss']
    y_label_fname_stubs = ['energy', 'force', 'loss']
    stats_dict = {}
    # 1. SOAP overlap
    if config_mean_best_soap_sims is not None:
        config_mean_best_soap_sims = np.array([config_mean_best_soap_sims[config_id] for config_id, _ in delta_energy_errors])
        plot_settings = PlotSettings(x_label='SOAP Similarity', y_label='\u2206 Loss', show_legend=False)
        correlation_scatter_plot(config_mean_best_soap_sims,
                                [delta_loss_values],
                                ['Influence'],
                                save_fp=f"{PLOT_DIR}/correlation/{save_name}_{test_set_name}_loss_influence_soap_corr.png",
                                plot_settings=plot_settings,
                                normalise=False)
        stats_dict['SOAP Sim Loss Corr.'] = np.corrcoef(config_mean_best_soap_sims, delta_loss_values)[0, 1]
    
    # 2. MACE desc overlap
    if config_mean_best_mace_sims is not None:
        config_mean_best_mace_sims = np.array([config_mean_best_mace_sims[config_id] for config_id, _ in delta_energy_errors])
        plot_settings = PlotSettings(x_label='MACE Similarity', y_label='\u2206 Loss', show_legend=False)
        correlation_scatter_plot(config_mean_best_mace_sims,
                                [delta_loss_values],
                                ['Influence'],
                                save_fp=f"{PLOT_DIR}/correlation/{save_name}_{test_set_name}_loss_influence_mace_corr.png",
                                plot_settings=plot_settings,
                                normalise=False)
        stats_dict['MACE Sim Loss Corr.'] = np.corrcoef(config_mean_best_mace_sims, delta_loss_values)[0, 1]

    # 3. Reaction Coordinate Distance
    train_ts_distances = np.array([train_ts_distances[config_id] for config_id, _ in delta_energy_errors])
    plot_settings = PlotSettings(x_label='TS Distance', y_label='\u2206 Loss', show_legend=False)
    correlation_scatter_plot(train_ts_distances,
                            [delta_loss_values],
                            ['Influence'],
                            save_fp=f"{PLOT_DIR}/correlation/{save_name}_{test_set_name}_loss_influence_ts_dist_corr.png",
                            plot_settings=plot_settings,
                            normalise=False)
    stats_dict['TS Dist. Loss Corr.'] = np.corrcoef(train_ts_distances, delta_loss_values)[0, 1]

    # 4a. Non-equilibrium structures (high energy outlying ground truth)
    # train_true_energy_deltas = np.array([train_true_energy_deltas[config_id] for config_id, _ in delta_energy_errors])
    # plot_influence_correlations(train_true_energy_deltas,
    #                             [delta_energy_error_values, delta_force_error_values, delta_loss_values],
    #                             save_name,
    #                             test_set_name,
    #                             y_labels,
    #                             y_label_fname_stubs=y_label_fname_stubs,
    #                             analysis_name='ground_truth_rel_energy',
    #                             x_label='Relative Energy (meV / atom)'
    #                             )
    # stats_dict['Ground Truth Rel. Energy Loss Corr.'] = np.corrcoef(train_true_energy_deltas, delta_loss_values)[0, 1]

    # 4 Non-equilibrium structures (high energy outlying predictions)
    train_pred_energy_deltas = np.array([train_pred_energy_deltas[config_id] for config_id, _ in delta_energy_errors])
    plot_settings = PlotSettings(x_label='Relative Energy (meV / atom)', y_label='\u2206 Loss', show_legend=False)
    correlation_scatter_plot(train_pred_energy_deltas,
                            [delta_loss_values],
                            ['Influence'],
                            save_fp=f"{PLOT_DIR}/correlation/{save_name}_{test_set_name}_loss_influence_pred_rel_energy_corr.png",
                            plot_settings=plot_settings,
                            normalise=False)
    stats_dict['Pred. Rel. Energy Loss Corr.'] = np.corrcoef(train_pred_energy_deltas, delta_loss_values)[0, 1]
        
    # 5. 2D plot of react coords and colour by loss
    r1_dists = [r1_dists[config_id] for config_id, _ in delta_energy_errors]
    r2_dists = [r2_dists[config_id] for config_id, _ in delta_energy_errors]
    points = np.array([np.array([x, y]) for x, y in zip(r1_dists, r2_dists)])
    plot_settings = PlotSettings(x_label='r1 / A', y_label='r2 / A')
    plot_2d([points],
            f"{PLOT_DIR}/2D/{save_name}_{test_set_name}_2D_react_coords_loss_influence.png", 
            colour_maps=[delta_loss_values],
            plot_settings=plot_settings,
            ref_points=[ref_ts_point],
            ref_point_labels=['TS'],
            colour_bar_label='\u2206 Loss'
            )
        
    # 6. SOAP + MACE PCA ANALYSIS
    proponent_ids = [config_id for config_id, _ in ranked_delta_losses[0:10]]
    opponent_ids = [config_id for config_id, _ in ranked_delta_losses[-10:]]
    soap_pca_1, soap_pca_2, mace_pca_1, mace_pca_2 = None, None, None, None
    plot_settings = PlotSettings(x_label='PC1', y_label='PC2')
    if train_soap_desc_2D is not None:
        # concat_pca_soap_descs = np.concatenate(train_config_pca_soap_descs, axis=0)
        soap_pca_1, soap_pca_2 = train_soap_desc_2D[:, 0], train_soap_desc_2D[:, 1]
    
        # TODO: pretty sure you need to average the SOAP descriptors here if you want to plot each config against influence!!!
        top_10_proponent_pca = np.array([train_soap_desc_2D[config_id, :] for config_id in proponent_ids])
        top_10_opponent_pca = np.array([train_soap_desc_2D[config_id, :] for config_id in opponent_ids])

        # [concat_pca_soap_descs, top_10_proponent_pca, top_10_opponent_pca],
        # colour_maps=[delta_loss_values, [0.0 for _ in range(len(delta_loss_values))], [1.0 for _ in range(len(delta_loss_values))]],

        plot_2d([train_soap_desc_2D],
                f"{PLOT_DIR}/2D/{save_name}_{test_set_name}_soap_PCA_influence_plot.png", 
                colour_maps=[delta_loss_values],
                plot_settings=plot_settings,
                colour_bar_label='\u2206 Loss'
                )
    
        stats_dict['SOAP PC1 Loss Corr.'] = np.corrcoef(soap_pca_1, delta_loss_values)[0, 1]

        # properties = {"PCA-1":soap_pca_1, "PCA-2":soap_pca_2, "score": delta_loss_values}
        # widget = chemiscope.show(train_configs, properties=properties)
        # widget.save("train_ds_influences.json")
    
    if train_config_pca_mace_descs is not None:
        mace_pca_1, mace_pca_2 = np.concatenate(train_config_pca_mace_descs, axis=0)[0, :], np.concatenate(train_config_pca_mace_descs, axis=0)[1, :]
        plot_2d([train_config_pca_mace_descs],
                f"{PLOT_DIR}/2D/{save_name}_{test_set_name}_mace_PCA_influence_plot.png", 
                colour_maps=[delta_loss_values],
                plot_settings=plot_settings,
                colour_bar_label='\u2206 Loss'
                )
    
    logger.info("===== STATS =====")
    for name, r in stats_dict.items():
        logger.info(f"{name}, r: {r}")
    logger.info("=================")

    # save csv of all features
    csv_save_name = f"{CSV_DIR}/{save_name}_{test_set_name}_feature_influence.csv"
    with open(csv_save_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Train Config', 'Base Model', 
                         'SOAP Pretrain Similarity', 'SOAP PC1', 'SOAP PC2', 
                         'MACE Pretrain Similarity', 'MACE PC1', 'MACE PC2',
                         'TS Dist', 'Ground Truth Rel. Energy', 'Pred. Rel. Energy',
                         'Energy Influence', 'Force Influence', 'Loss Influence'])

        csv_tuples = []
        for i, config in enumerate(train_configs):
            csv_tuples.append((i, 
                               base_model_name, 
                               config_mean_best_soap_sims[i] if config_mean_best_soap_sims is not None else 'None', 
                               soap_pca_1[i] if soap_pca_1 is not None else 'None', 
                               soap_pca_2[i] if soap_pca_2 is not None else 'None', 
                               config_mean_best_mace_sims[i] if config_mean_best_mace_sims is not None else 'None', 
                               mace_pca_1[i] if mace_pca_1 is not None else 'None', 
                               mace_pca_2[i] if mace_pca_2 is not None else 'None', 
                               train_ts_distances[i], train_true_energy_deltas[i], train_pred_energy_deltas[i], 
                               delta_energy_error_values[i], delta_force_error_values[i], delta_loss_values[i]))

        writer.writerows(csv_tuples)
        logger.info(f"Saved results to {csv_save_name}")


if __name__ == "__main__":

    overall_start_time = time()
    
    # ===== SPECIFY MODELS TO COMPARE HERE =====
    base_model_name, ref_pretrain_ds = "MACE-OFF23_medium", f"{DATASETS_DIR}/spice/spice_train_sample_n=10000.xyz"
    # base_model_name, ref_pretrain_ds = "GO-MACE-23", f"{DATASETS_DIR}/go_mace/GO-MACE-23_train_sample_n=3013.xyz"
    # base_model_name, ref_pretrain_ds = "MACE-MP0-128-L1", f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_sample_n=10000.xyz"

    # ===== SPECIFY THE LOSS =====
    energy_weight, forces_weight = 1.0, 10.0
    loss_func = WeightedEnergyForceMSELoss

    # ===== SELECT TRAIN DS AND RUN NAMES =====
    train_ds_name = 'endo_DA_train_size_sorted'
    run_name = f'100_epochs_{train_ds_name}_single_config'
    
    # ===== SELECT SUBSET FOR DEBUG =====
    model_subset = (0, 100)
    
    # ===== SPECIFY TEST SETS PARAMETERS HERE =====

    # === endo DA full test set ===
    test_name = "endo_DA_test"
    test_set_fpaths = [
        f"{DATASETS_DIR}/da_paper_data/valid/endo_ace_wB97M_nonpca_orca.xyz",
        f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz",
        f"{TRAIN_SUBSETS_DIR}/endo_DA_valid_size_sorted.xyz",
        f"{DATASETS_DIR}/da_paper_data/neb/endo_DA_neb_gas_wB97M.xyz", 
        # f"{DATASETS_DIR}/da_paper_data/neb/endo_DA_neb_water_wB97M.xyz"
        ]
    csv_category_names = ["Endo DA Vac Traj", "Endo DA Water Traj", "Endo DA Valid", "Endo DA NEB Gas", "Endo DA NEB Water"]
    custom_xyz_parsers = [None, None, None, None, autode_props_parser]
    test_elements = {'C', 'H', 'O'}
    ts_ref_config = flexible_ase_read(f'{ENDO_DA_DATA_DIR}/endo_DA_vac_TS_wB97M.xyz')[0]
    r1, r2 = (1, 12), (6, 11)

    # ===== SPECIFY ANALYSIS PARAMETERS =====
    soap_comparison = True
    soap_similarity_comparison = False
    mace_comparison = False
    mace_similarity_comparison = False
    desc_pca_dim = None #32
    # =======================================

    # get save name / model names
    # WARNING : model names here should be a 1:1 mapping to the configs in train_ds_name
    save_name = f"{base_model_name}_{run_name}_fine_tuned"
    model_name_stub = f"{base_model_name}_{run_name}*fine_tuned*"
    model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
    assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
    model_names = [get_name_from_path(model_fpath) for model_fpath in model_fpaths]
    model_names = model_names[model_subset[0]:model_subset[1]]  # TODO: DEBUG!!!

    # load cache file
    hdf5_file = load_hdf5_cache()

    # load train configs
    train_ds_fp = f"{TRAIN_SUBSETS_DIR}/{train_ds_name}.xyz"
    train_configs = flexible_ase_read(train_ds_fp) if model_subset is None else flexible_ase_read(train_ds_fp)[model_subset[0]: model_subset[1]]
    logger.info(f'Num Considered Train Configs: {len(train_configs)}')
    test_set_names = [get_name_from_path(test_set_fpath) for test_set_fpath in test_set_fpaths]
    
    train_config_pca_soap_descs, train_config_pca_mace_descs = None, None
    config_mean_best_soap_sims, config_mean_best_mace_sims = None, None
    if soap_comparison:

        # get SOAP descriptors of training set
        start_time = time()
        train_config_average_soap = get_soap_descriptors(train_configs, 
                                                         specific_elements=test_elements, 
                                                         average='outer', 
                                                         compress_dim=desc_pca_dim, 
                                                         flatten_output=True, 
                                                         cache_name=train_ds_name)
        logger.info(f"Loaded / calculated {len(train_configs)} average train SOAP descriptors in {time() - start_time:.4f}s...")

        # get 2D PCA of train SOAP descs (need to concatenate then reconstruct per-config descs)
        train_soap_desc_2D = run_pca(train_config_average_soap, n_components=2)

        if soap_similarity_comparison:
            # TODO: turn this into standalone function

            # train soap desc
            start_time = time()
            train_config_soap_vecs = get_soap_descriptors(train_configs, 
                                                          specific_elements=test_elements, 
                                                          average='off', 
                                                          compress_dim=desc_pca_dim, 
                                                          flatten_output=False, 
                                                          cache_name=train_ds_name)
            logger.info(f"Loaded / calculated {len(train_configs)} train SOAP descriptors in {time() - start_time:.4f}s...")

            # pretrain SOAP desc
            start_time = time()
            pretrain_configs = flexible_ase_read(ref_pretrain_ds)
            pretrain_soap_mat = get_soap_descriptors(pretrain_configs, specific_elements=test_elements, average='off', 
                                                    compress_dim=desc_pca_dim, flatten_output=True, cache_name=get_name_from_path(ref_pretrain_ds), n_jobs=N_CORES)
            logger.info(f"Loaded / calculated {len(pretrain_configs)} pretrain SOAP descriptors in {time() - start_time:.4f}s...")

            # get best cosine similarities of each x1 -> x2 descriptor
            start_time = time()
            cos_sim_cache_fpath = f"{NPZ_DIR}/{get_name_from_path(ref_pretrain_ds)}_{train_ds_name}_mean_best_soap_cos_sims.npz"
            if os.path.exists(cos_sim_cache_fpath):
                cache_data = load_data_cache(cos_sim_cache_fpath)
                config_mean_best_soap_sims = cache_data['mean_best_cos_sims'].tolist()
                logger.info(f"Loaded mean best SOAP cosine similarities from {cos_sim_cache_fpath}...")   
            else:
                logger.info(f'Calculating best SOAP cosine similarities between {get_name_from_path(ref_pretrain_ds)} and {train_ds_name}...')
                config_mean_best_soap_sims = []
                for config_soap_vec in train_config_soap_vecs:
                    best_sim_tuples, best_sim_mean, _ = calc_best_similarities(config_soap_vec, pretrain_soap_mat)
                    config_mean_best_soap_sims.append(best_sim_mean)
                save_npz_data_cache(cos_sim_cache_fpath, {'mean_best_cos_sims': np.array(config_mean_best_soap_sims)})
            logger.info(f"Loaded / calculated {len(config_mean_best_soap_sims)} cosine similarities in {time() - start_time:.4f}s...")

    if mace_comparison:   

        calc = load_model_calculator(base_model_name, model_loader='MACE')
        
        # train MACE desc
        train_cache_name = f'{base_model_name}_' + get_name_from_path(train_ds_fp)
        train_config_mace_descs = get_mace_descriptors(calc,
                                                       train_configs,
                                                       invariants_only=True,
                                                       compress_dim=None,
                                                       flatten_output=False,
                                                       cache_name=train_cache_name)
        
        # get 2D PCA of train MACE descs
        mace_desc = np.concatenate(train_config_mace_descs, axis=0)
        mace_desc = run_pca(mace_desc, n_components=2)   
        train_config_pca_mace_descs = unflatten_array_by_lengths(mace_desc, [len(config) for config in train_configs])

        if mace_similarity_comparison:
            # pretrain MACE desc
            pretrain_cache_name = f'{base_model_name}_' + get_name_from_path(ref_pretrain_ds)
            pretrain_configs = flexible_ase_read(ref_pretrain_ds)
            pretrain_mace_desc = get_mace_descriptors(calc,
                                                    pretrain_configs,
                                                    invariants_only=True,
                                                    compress_dim=None,
                                                    flatten_output = True,
                                                    cache_name = pretrain_cache_name)   

            # get best cosine similarities of each x1 -> x2 descriptor
            cos_sim_cache_fpath = f"{NPZ_DIR}/{get_name_from_path(ref_pretrain_ds)}_{train_ds_name}_{base_model_name}_mean_best_mace_cos_sims.npz"
            if os.path.exists(cos_sim_cache_fpath):
                cache_data = load_data_cache(cos_sim_cache_fpath)
                config_mean_best_mace_sims = cache_data['mean_best_cos_sims'].tolist()
                logger.info(f"Loaded mean best MACE cosine similarities from {cos_sim_cache_fpath}...")   
            else:
                logger.info(f'Calculating best MACE cosine similarities between {get_name_from_path(ref_pretrain_ds)} and {train_ds_name}...')
                config_mean_best_mace_sims = []
                for i, config_mace_desc in enumerate(train_config_mace_descs):
                    best_sim_tuples, best_sim_mean, _ = calc_best_similarities(config_mace_desc, pretrain_mace_desc)
                    config_mean_best_mace_sims.append(best_sim_mean)
                    if i % 10 == 0:
                        logger.info(f"Calculated best MACE cos sims for {i} / {len(train_config_mace_descs)}...")
                save_npz_data_cache(cos_sim_cache_fpath, {'mean_best_cos_sims': np.array(config_mean_best_mace_sims)})
    
    # calculate deviations of training samples from the ts
    ts_ref_bond_distances = {r1: ts_ref_config.get_distance(*r1), r2: ts_ref_config.get_distance(*r2)}
    ts_ref_bond_dist_tuple = tuple([dist for dist in ts_ref_bond_distances.values()])
    logger.info('TS ref bond distances: ', ts_ref_bond_distances)
    r1_dists = [config.get_distance(*r1, mic=True) for config in train_configs]
    r2_dists = [config.get_distance(*r2, mic=True) for config in train_configs]
    train_ts_distances = np.array([abs(r1_dists[i] - ts_ref_bond_distances[r1]) + abs(r2_dists[i] - 
                          ts_ref_bond_distances[r2]) for i in range(len(train_configs))])
    
    # get list of indices of ts distances to filter out configs based on element
    # for i, config in enumerate(train_configs):
    #     assert config[r1[0]].symbol == 'C' and config[r1[1]].symbol == 'C'
    #     assert config[r2[0]].symbol == 'C' and config[r2[1]].symbol == 'C'
    # r1_atoms, r2_atoms = [3, 14], [8, 13] # reaction coordinates in water

    # get energy / force deltas

    # group configs by number of atoms
    atom_num_groups = {}
    for config in train_configs:
        if atom_num_groups.get(len(config)) is None:
            atom_num_groups[len(config)] = [config]
        else:
            atom_num_groups[len(config)].append(config)

    atom_num_group_energy_mins = {num_atoms: np.min(get_ase_props(group_configs, prop='energy')) 
                                  for num_atoms, group_configs in atom_num_groups.items()}
    atom_num_group_force_mins = {num_atoms: np.min(get_ase_props(group_configs, prop='forces')) # TODO: add option to flatten output
                                 for num_atoms, group_configs in atom_num_groups.items()}
    ground_truth_energy_deltas = [abs(get_ase_atoms_energy(config) - atom_num_group_energy_mins[len(config)]) for config in train_configs]
    # test_force_deltas = [abs(get_ase_atoms_energy(config) - atom_num_group_mins[len(config)]) for config in train_configs]

    # get training set predicted values
    base_pred_train_energies_list, base_pred_train_forces_list = get_model_predictions(base_model_name,
                                                                                       [train_ds_fp],
                                                                                       custom_xyz_parsers=custom_xyz_parsers,
                                                                                       hdf5_file=hdf5_file)
    base_pred_train_energies, base_pred_train_forces = base_pred_train_energies_list[0], base_pred_train_forces_list[0]
    
    # group config ids by number of atoms
    atom_num_groups = {}
    for i, config in enumerate(train_configs):
        if atom_num_groups.get(len(config)) is None:
            atom_num_groups[len(config)] = [i]
        else:
            atom_num_groups[len(config)].append(i)

    atom_num_group_pred_energy_mins = {num_atoms: np.min([base_pred_train_energies[i] for i in group_indices]) 
                                       for num_atoms, group_indices in atom_num_groups.items()}
    pretrain_pred_energy_deltas = [abs(base_pred_train_energies[i] - atom_num_group_pred_energy_mins[len(config)]) for i, config in enumerate(train_configs)]
    
    # forces
    # get forces as per config list of arrays
    # pred_config_train_forces = unflatten_array_by_lengths(pred_train_forces, [len(config) for config in train_configs])
    # # atom_num_group_force_mins = {num_atoms: np.min([pred_config_train_forces[i] for i in group_indices]) # TODO: add option to flatten output
    #                              for num_atoms, group_indices in atom_num_groups.items()}
    # pretrain_pred_force_deltas = [abs(get_ase_atoms_energy(config) - atom_num_group_energy_mins[len(config)]) for config in train_configs]


    # get base model predictions
    start_time = time()
    base_pred_energies, base_pred_forces = get_model_predictions(base_model_name,
                                                                           test_set_fpaths,
                                                                           custom_xyz_parsers=custom_xyz_parsers,
                                                                           hdf5_file=hdf5_file)

    # get test set energies / forces and run inference on test set with models
    all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data(model_names,
                                                                                                              test_set_fpaths,
                                                                                                              custom_xyz_parsers=custom_xyz_parsers,
                                                                                                              hdf5_file=hdf5_file)
    logger.info(f"Loaded / calculated model data in {time() - start_time:.4f}s...")

    # do individual test set analyses
    # for i, (test_energies, test_forces, test_set_fpath) in enumerate(zip(all_test_energies, all_test_forces, test_set_fpaths)):

    #     test_configs = flexible_ase_read(test_set_fpath)
    #     pred_energies_dict = {model_name: all_pred_energies[i] for model_name, all_pred_energies in all_pred_energies_dict.items()}
    #     pred_forces_dict = {model_name: all_pred_forces[i] for model_name, all_pred_forces in all_pred_forces_dict.items()}
    #     test_set_name = test_set_names[i]
    #     influence_analysis(test_configs,
    #                        test_energies,
    #                        test_forces,
    #                        pred_energies_dict,
    #                        pred_forces_dict,
    #                        base_pred_energies[i],
    #                        base_pred_forces[i],
    #                        model_names,
    #                        train_configs,   
    #                        train_ds_name,
    #                        test_set_name,
    #                        save_name,
    #                        config_mean_best_soap_sims,
    #                        config_mean_best_mace_sims,
    #                        train_config_pca_soap_descs,
    #                        train_config_pca_mace_descs,
    #                        train_ts_distances,
    #                        r1_dists,
    #                        r2_dists,
    #                        ts_ref_bond_dist_tuple,
    #                        ground_truth_energy_deltas, 
    #                        pretrain_pred_energy_deltas)   

    # analysis of combined test sets  
    # concatenate all test and pred values
    concat_test_configs = [config for test_set_fpath in test_set_fpaths for config in flexible_ase_read(test_set_fpath)]
    concat_test_energies = np.array([energy for test_energies in all_test_energies for energy in test_energies])
    concat_test_forces = np.array([force for test_forces in all_test_forces for force in test_forces])
    concat_base_pred_energies = np.array([energy for pred_energies in base_pred_energies for energy in pred_energies])
    concat_base_pred_forces = np.array([force for pred_forces in base_pred_forces for force in pred_forces])

    concat_pred_energies_dict = {}
    concat_pred_forces_dict = {}
    for model_name in all_pred_energies_dict.keys():
        all_pred_energies = all_pred_energies_dict[model_name]
        all_pred_forces = all_pred_forces_dict[model_name]
        concat_pred_energies_dict[model_name] = np.array([energy for pred_energies in all_pred_energies for energy in pred_energies])
        concat_pred_forces_dict[model_name] = np.array([force for pred_forces in all_pred_forces for force in pred_forces])

    test_set_name = 'overall'
    influence_analysis(concat_test_configs,
                       concat_test_energies,
                       concat_test_forces,
                       concat_pred_energies_dict,
                       concat_pred_forces_dict,
                       concat_base_pred_energies, 
                       concat_base_pred_forces,
                       model_names,
                       train_configs,   
                       train_ds_name,
                       test_set_name,
                       save_name,
                       config_mean_best_soap_sims,
                       config_mean_best_mace_sims,
                       train_soap_desc_2D,
                       train_config_pca_mace_descs,
                       train_ts_distances,
                       r1_dists,
                       r2_dists,
                       ts_ref_bond_dist_tuple,
                       ground_truth_energy_deltas, 
                       pretrain_pred_energy_deltas,
                       proponents_only=False)   

    logger.info(f"Ran influence analysis in {time() - overall_start_time:.4f}s...")
