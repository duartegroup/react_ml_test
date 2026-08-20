"""
Test scripts for react ml test repo.
"""
import os
import unittest
import sys

from react_ml_test.helpers.sampling import sample_dataset


sys.path.append("../")
from mace.calculators import MACECalculator
from ase.build import molecule
from ase import units
from ase.md.langevin import Langevin
from ase.io import read
from react_ml_test.helpers.sorting import sort_configs_by_cv_dist, sort_configs_by_soap_desc
from react_ml_test.helpers.plot import correlation_scatter_plot
from react_ml_test.helpers.traj_eval import atom_distance_plot, print_atom_distance_table
from react_ml_test.helpers.dynamics import get_trajectory_with_mlptrain
from react_ml_test.helpers.settings import DATASETS_DIR, MD_INPUT_DIR, MODEL_DIR, OUT_DIR, TEST_DATA_DIR, TEST_DIR, TEST_PLT_DIR, TEST_TRJ_DIR, TRAIN_SUBSETS_DIR
from react_ml_test.helpers.utils import calc_energy_shifts, eV_to_meV_per_atom, guess_per_element_mean_energies, load_model_calculator
from react_ml_test.helpers.model_eval import get_model_predictions, get_test_and_pred_data
from react_ml_test.helpers.descriptors import get_soap_descriptors
from react_ml_test.helpers.dim_red import calc_best_similarities, run_umap
from react_ml_test.helpers.io import flexible_ase_read, get_name_from_path, get_xyz_config_offsets, load_data_cache, save_dataset_batches, save_npz_data_cache
from react_ml_test.helpers.stats import calculate_comp_rmse, calculate_mae, calculate_rmse
import glob
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
 

class PlotTests(unittest.TestCase):

    def test_atom_distance_plot(self):

        r1_atoms = [1, 12]  # reactant 1 atoms
        r2_atoms = [6, 11]  # reactant 2 atoms

        # TODO:
        # specify detection bonds for different products
        threshold = 2.0
        prod_def_R = {'name': 'R', 'bond_defs': [(1, 12, False, threshold), (6, 11, False, threshold)]}
        prod_def_P = {'name': 'P', 'bond_defs': [(1, 12, True, threshold), (6, 11, True, threshold)]}
        prod_def_S1 = {'name': 'S1', 'bond_defs': [(2, 17, True, threshold)]}
        # prod_def_S2 = {'name': 'S2', 'bond_defs': [(1, 12, True, threshold), (6, 11, True, threshold)]}

        r1_range = (1.0, 3.0)
        r2_range = (1.0, 5.0)

        # vacuum
        trajectories = []
        fpaths = sorted(glob.glob(f"{TEST_TRJ_DIR}/*"))
        for fpath in fpaths:
            print(f"Loading file {fpath}...")
            trajectories.append(flexible_ase_read(fpath))

        atom_distance_plot(trajectories, 
                           r1_atoms, 
                           r2_atoms, 
                           f"{TEST_PLT_DIR}/atom_distance_test.png",
                           r1_range=r1_range,
                           r2_range=r2_range,
                           prod_defs=[prod_def_R, prod_def_P, prod_def_S1]
                           )
        
        # periodic solvent

    

class TestPredictions(unittest.TestCase):

    def test_foundation_model_al(self):

        # TODO!

        import mlptrain as mlt
        from mlptrain.configurations.explicit_solvation import generate_init_solv_configs

        # TODO: load system, here with explicit water etc.
        water_mol = mlt.Molecule(name=f'{MD_INPUT_DIR}/h2o.xyz')
        ts_fp = f'{MD_INPUT_DIR}/cis_endo_TS_wB97M.xyz'
        ts_mol = mlt.Molecule(name=ts_fp)

        # create system in large box with 40 water molecules (without coords)
        ts_in_water = mlt.System(ts_mol, box=mlt.Box([100, 100, 100]))
        ts_in_water.add_molecules(water_mol, num=40)

        # generate random configs of TS in water
        TS_info = (ts_fp, 0, 1)
        # ts_in_water_init = generate_init_solv_configs(n=10, solvent_mol=water_mol, bulk_solvent=True, include_TS=True, TS_info=TS_info)

        model_name = "MACE-OFF23_small"
        mace_model = mlt.potentials.MACE(model_name, system=ts_in_water)

    def test_explicit_water_mlp_dynamics(self):
        import mlptrain as mlt

        # load system, here with explicit water etc.
        water_mol = mlt.Molecule(name=f'{MD_INPUT_DIR}/h2o.xyz')
        ts_fp = f'{MD_INPUT_DIR}/cis_endo_TS_wB97M.xyz'
        ts_mol = mlt.Molecule(name=ts_fp)

        # create system in large box with 40 water molecules (without coords)
        ts_in_water = mlt.System(ts_mol, box=mlt.Box([100, 100, 100]))
        ts_in_water.add_molecules(water_mol, num=40)

        model_name = "MACE-OFF23_small"
        mace_model = mlt.potentials.MACE(model_name, system=ts_in_water)

        get_trajectory_with_mlptrain(ts_in_water.random_configuration(), mace_model, f'DA_explicit_{model_name}')

    def test_foundation_ase_eval(self):

        # define structure
        structure = molecule("H2O")

        # load calculator
        calc = MACECalculator(model_path="../models/MACE-OFF23_small.model", device='cuda')

        # to run the dynamics, we:
        # 1. attach the calculator to the Atoms object
        structure.set_calculator(calc)

        # 2. create the dynamics object
        dynamics = Langevin(
            structure,
            timestep=0.1 * units.fs,
            temperature_K=300,
            friction=0.001 / units.fs,
        )

        # re-write trajectory file if it doesn't exist already
        traj_fp = "out/trajectory_ase.xyz"
        if not os.path.isdir("data"):
            os.mkdir("data")
        if os.path.exists(traj_fp):
            os.remove(traj_fp)

        # 3. attach a callback to periodically write the trajectory to a file
        dynamics.attach(
            lambda: structure.write(traj_fp, append=True),
            interval=10,
        )

        # 4. actually run the dynamics
        print("Running dynamics trajectory...")
        dynamics.run(steps=1000)  # 10,000 steps of 0.1 fs each is 1 ps

        trajectory = read("out/trajectory_ase.xyz", index=":")
        assert isinstance(trajectory, list)
        print("Number of trajectory frames: ", len(trajectory))

        # create atom distance plot between h atoms
        atom_distance_plot(trajectory, (1, 2))

    def test_foundation_lammps_eval(self):

        # TODO: this!!!

        from load_atoms import load_dataset

        # this will take ~2 minutes the first time,
        # then ~1 second after that
        amorphous_dataset = load_dataset("C-SYNTH-23M")
        subset = amorphous_dataset.filter_by(anneal_T=2_000, density=2.5)
        structure = subset[-10]

        # write to a format suitable for LAMMPS
        structure.write("data/md_inputs/start-structure.data", format="lammps-data")

        from graph_pes.models import load_model
        from graph_pes.deploy import deploy_model
        from pathlib import Path

        # deploy_path = Path("md-input") / "schnet-cgap17.lammps-model"
        # normal_model = load_model("resources/schnet-cgap17.pt")
        # deploy_model(normal_model, path=deploy_path)

        # TODO: here get thing going

        final_structure = read("md-output/dumps/1000.data")

    def test_eval_trajectory(self):

        trajectory = read("out/trajectory_ase.xyz", index=":")

        print("Trajectory: ", trajectory)
        # create atom distance plot between h atoms
        atom_distance_plot(trajectory, (1, 2))


    def test_multi_correlation_plot(self):

        # test_set = flexible_ase_read(f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_data_test.xyz")
        test_set = flexible_ase_read(f"{TEST_DATA_DIR}/structures/endo2_water_data_test.xyz")
        offset_ref_fpath = f"{TEST_DATA_DIR}/structures/endo_DA_test.xyz"
        true_energies = [c.get_potential_energy() for c in test_set]
        true_per_element_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, dtype=np.float64)

        # model_names = ["MACE-OFF23_small", "GO-MACE-23", "ani-1xnr"]
        # model_names = ["MACE-OFF23_medium_endo_DA_1.0_train_from_scratch", "MACE-OFF23_medium"]
        # model_names = ["MACE-OFF23_small", "GO-MACE-23", "ani-1xnr"]
        model_names = ["ani500k_large_CC"]
        # model_names = ["MACE-OFF23_medium_endo_DA_1.0_train_from_scratch", "MACE-ANI-CC", "GO-MACE-23", "ani-1xnr"]
        
        # get true and pred energy shifts
        true_energy_shifts = calc_energy_shifts(test_set, true_per_element_energy_offsets)
        shifted_true_energies = true_energies - true_energy_shifts
        shifted_true_energies_meV_per_atom = np.array([eV_to_meV_per_atom(energy, len(config)) for energy, config in zip(shifted_true_energies, test_set)])

        model_pred_energies = []
        for model_name in model_names:
            calculator = load_model_calculator(model_name, device="cuda:0")
            pred_per_element_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, calc=calculator, calc_name=model_name, dtype=np.float64)
            pred_energy_shifts = calc_energy_shifts(test_set, pred_per_element_energy_offsets)

            # print("Pred Energy Shifts: ", pred_energy_shifts)

            # 
            pred_energies, config_lens = [], []
            for config in test_set:
                config.calc = calculator
                energy = config.get_potential_energy()
                pred_energies.append(energy)

                config_lens.append(len(config))

            shifted_pred_energies = pred_energies - pred_energy_shifts

            shifted_pred_energies_meV_per_atom = np.array([eV_to_meV_per_atom(energy, config_len) for energy, config_len in zip(shifted_pred_energies, config_lens)])
            model_pred_energies.append(shifted_pred_energies_meV_per_atom)

            # print("Correct method energies: ", shifted_pred_energies_meV_per_atom[0:10])
        
        correlation_scatter_plot(shifted_true_energies_meV_per_atom,
                                 model_pred_energies,
                                 model_names,
                                 f"{TEST_PLT_DIR}/correlation_test.png",
                                 x_label='Pred. Norm E meV / atom',
                                 y_label='DFT Norm E meV / atom',
                                 )
        
        # all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data(model_names,
        #                                                                                                           [f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_data_test.xyz"],
        #                                                                                                           offset_ref_fpath=offset_ref_fpath,
        #                                                                                                           model_loaders=None,
        #                                                                                                           pbc=False)
        
        # print("Main method ANI energies: ", all_pred_energies_dict['ani-1xnr'][0])
        # correlation_scatter_plot(all_test_energies[0],
        #                          [pred_energies[0] for pred_energies  in all_pred_energies_dict.values()],
        #                          [model_name for model_name in all_pred_energies_dict.keys()],
        #                          f"{TEST_PLT_DIR}/correlation_test_main_method.png",
        #                          x_label='meV / atom',
        #                          y_label='meV / atom',
        #                          )

        return


    def test_energy_shifting(self):
        """
        Test to see how energy shifting works over datasets with configs of different sizes.
        """

        # multi_size = flexible_ase_read(f"{TEST_DATA_DIR}/structures/multi_size.xyz")
        # multi_size = flexible_ase_read(f"{TEST_DATA_DIR}/structures/endo_in_water_random_data_test.xyz")

        test_set_fpaths = [
                    f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_data_test.xyz",
                    f"{TEST_DATA_DIR}/structures/watertol_ace_wB97M_27_data_test.xyz",
                    f"{TEST_DATA_DIR}/structures/endo2_water_data_test.xyz",
                    f"{TEST_DATA_DIR}/structures/endo_in_water_random_data_test.xyz",
                    f"{TEST_DATA_DIR}/structures/endo_DA_test.xyz",
                    ]
        
        # test_set_fpaths = [f"{TEST_DATA_DIR}/structures/watertol_ace_wB97M_27_data_test.xyz", 
        #                    f"{TEST_DATA_DIR}/structures/endo_DA_test.xyz"]

        offset_ref_fpath = f"{TEST_DATA_DIR}/structures/endo_DA_test.xyz"
        true_per_element_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, dtype=np.float64)
        print(f"True Per-element means: {true_per_element_energy_offsets}")

        rmses_eV, rmses_meV_atom, num_samples = [], [], []
        for fpath in test_set_fpaths: 
            test_configs = flexible_ase_read(fpath)
            true_energies = np.array([c.get_potential_energy() for c in test_configs])
            
            model_name, model_loader = "ani-1xnr", "ani"
            # model_name, model_loader = "GO-MACE-23", "MACE"
            calculator = load_model_calculator(model_name, device="cuda:0", model_loader=model_loader)
            
            # get true and pred energy shifts
            pred_per_element_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, calc=calculator, calc_name=model_name, dtype=np.float64)
            true_energy_shifts = calc_energy_shifts(test_configs, true_per_element_energy_offsets)
            pred_energy_shifts = calc_energy_shifts(test_configs, pred_per_element_energy_offsets)
            print(f"Model: {model_name} Per-element means: {pred_per_element_energy_offsets}")

            # 
            pred_energies, config_lens = [], []
            for config in test_configs:
                config.calc = calculator
                energy = config.get_potential_energy()
                if model_loader == "ani":
                    energy /= units.Hartree
                pred_energies.append(energy)
                config_lens.append(len(config))

            pred_energies = np.array(pred_energies)

            shifted_true_energies = true_energies - true_energy_shifts
            shifted_pred_energies = pred_energies - pred_energy_shifts

            shifted_true_energies_meV_per_atom = np.array([eV_to_meV_per_atom(energy, len(config)) for energy, config in zip(shifted_true_energies, test_configs)])
            shifted_pred_energies_meV_per_atom = np.array([eV_to_meV_per_atom(energy, len(config)) for energy, config in zip(shifted_pred_energies, test_configs)])

            # print("True Energies: ", true_energies[0:10])
            # print("True Energy Shifts: ", true_energy_shifts[0:10])
            # print("Pred Energies: ", pred_energies[0:10])
            # print("Pred Energy Shifts: ", pred_energy_shifts[0:10])
            # print("Shifted True Energies eV: ", shifted_true_energies[0:10])
            # print("Shifted True Energies meV / atom: ", shifted_true_energies_meV_per_atom[0:10])
            # print("SECOND Shifted True Energies meV / atom: ", second_shifted_true_energies_meV_per_atom[0:10])
            # # print("Shifted Pred Energies eV: ", shifted_pred_energies[0:10])
            # print("Shifted Pred Energies meV / atom: ", shifted_pred_energies_meV_per_atom[0:10])
            # print("SECOND Shifted Pred Energies meV / atom: ", second_shifted_pred_energies_meV_per_atom[0:10])

            mae = calculate_mae(shifted_pred_energies, shifted_true_energies)
            mae_meV_per_atom = calculate_mae(shifted_pred_energies_meV_per_atom, shifted_true_energies_meV_per_atom)
            rmse = calculate_rmse(shifted_pred_energies, shifted_true_energies)
            rmse_meV_per_atom = calculate_rmse(shifted_pred_energies_meV_per_atom, shifted_true_energies_meV_per_atom)
            print("TEST SET: ", fpath)
            print("MAE eV: ", mae)
            print("MAE meV / atom: ", mae_meV_per_atom)
            print("RMSE eV: ", rmse)
            print("RMSE meV / atom: ", rmse_meV_per_atom)

            # rmses_eV.append(rmse)
            # rmses_meV_atom.append(rmse_meV_per_atom)
            # num_samples.append(len(test_configs))

        # total_rmse, total_num_samples = rmses_eV[-1], num_samples[-1]
        # rmses_eV, num_samples = rmses_eV[:-1], num_samples[:-1]
        # print("Rmses eV: ", rmses_eV)
        # print("Num Samples: ", num_samples)

        # TODO: FIX! need to store MSEs and then sqrt
        # weighted_rmse = np.sqrt(sum(np.sum(np.square(err)) for err in errors) / sum(len(err) for err in errors))
        # weighted_rmse = sum([rmse_val * n for rmse_val, n in zip(rmses_eV, num_samples)]) / float(sum([n for n in num_samples]))
        # assert total_num_samples == sum(num_samples)
        # assert np.isclose(total_rmse, weighted_rmse, atol=1e-10), f"Total RMSE: {total_rmse}, Weighted RMSE: {weighted_rmse}"

    def test_energy_average_offset(self):
        """
        Test that our numpy based average elemental offset code is the same as in graph-pes.
        """

        from graph_pes.atomic_graph import AtomicGraph, to_batch
        from graph_pes.interfaces import go_mace_23
        from graph_pes.utils.shift_and_scale import guess_per_element_mean_and_var

        # fake energy example
        h2o_mult = flexible_ase_read(f"{TEST_DATA_DIR}/structures/h2o_multiple.xyz")
        print("\nCalculating true energy offsets: ")
        my_true_offsets = guess_per_element_mean_energies(h2o_mult, dtype=np.float32)
        true_energy_shifts = calc_energy_shifts(h2o_mult, dtype=np.float32)
        # print("True Energy Shifts: ", true_energy_shifts)

        model_name = "GO-MACE-23"
        calculator = load_model_calculator(model_name, device="cuda:0")

        print("Calculating pred energy offsets: ")
        my_pred_offsets = guess_per_element_mean_energies(h2o_mult, calc=calculator, calc_name=model_name, dtype=np.float32)
        pred_energy_shifts = calc_energy_shifts(h2o_mult, calc=calculator, calc_name=model_name, dtype=np.float32)
        # print("Pred Energy Shifts: ", pred_energy_shifts)
        
        # reload ase configs so the energies don't
        h2o_mult = flexible_ase_read(f"{TEST_DATA_DIR}/structures/h2o_multiple.xyz")

        # graph-pes
        test_graphs = [AtomicGraph.from_ase(h2o, cutoff=3.7) for h2o in h2o_mult]
        test_graph_batch = to_batch(test_graphs)

        print("\nCalculating graph-pes true energy offsets: ")
        graph_pes_true_offsets, _ = guess_per_element_mean_and_var(
            test_graph_batch.properties["energy"], test_graph_batch
        )
        print("\nGraph PES True Offsets: ", graph_pes_true_offsets)

        model = go_mace_23(precision="float32")
        graph_pes_pred_energies = model.predict_energy(test_graph_batch)
        # graph_pes_pred_energies = model.get_energy_predictions(test_graph_batch)
        graph_pes_pred_offsets, _ = guess_per_element_mean_and_var(
            graph_pes_pred_energies.detach(), test_graph_batch
        )
        print("Graph PES Pred Offsets: ", graph_pes_pred_offsets)

        # assert offset calculations are the same
        true_mask = ~np.isclose(np.array(list(graph_pes_true_offsets.values())), np.array(list(my_true_offsets.values())), rtol=1e-4, atol=1e-4)
        pred_mask = ~np.isclose(np.array(list(graph_pes_pred_offsets.values())), np.array(list(my_pred_offsets.values())), rtol=1e-5, atol=1e-8)
        assert ~np.any(true_mask)
        assert ~np.any(pred_mask)

    def test_graph_pes_inference(self):
        
        from graph_pes.atomic_graph import AtomicGraph
        from graph_pes.interfaces import go_mace_23


        # load e.g. structure
        h2o = flexible_ase_read(f"{TEST_DATA_DIR}/structures/h2o.xyz")[0]
        
        # graph-pes
        test_graph = AtomicGraph.from_ase(h2o, cutoff=3.7)
        model = go_mace_23(precision=None)
        graph_pes_pred_energy = model.predict_energy(test_graph)
        graph_pes_pred_forces = model.predict_forces(test_graph)

        # now compare our package
        model_name = "GO-MACE-23"
        calculator = load_model_calculator(model_name, device="cuda:0")
        h2o.calc = calculator
        my_pred_energy = h2o.get_potential_energy()
        my_pred_forces = h2o.get_forces()

        print("Graph PES Pred Energy: ", graph_pes_pred_energy)
        print("My Pred Energy: ", my_pred_energy)

        print("Graph PES Pred Forces: ", graph_pes_pred_forces)
        print("My Pred Forces: ", my_pred_forces)

        assert(graph_pes_pred_energy.float() == my_pred_energy)


    def test_graph_pes_inference_endo_DA(self):
        
        from graph_pes.atomic_graph import AtomicGraph, to_batch
        from graph_pes.interfaces import go_mace_23
        from graph_pes.utils.analysis import parity_plot

        # load e.g. structure
        # test_set_fpath = f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_data_test.xyz"
        test_set_fpath = f"{TEST_DATA_DIR}/structures/endo_in_water_random_data_test.xyz"
        # test_set_fpath = f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_data_test.xyz"
        # test_set_fpath = f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_data_test.xyz"
        test_configs = flexible_ase_read(test_set_fpath)
        
        # graph-pes
        test_graphs = [AtomicGraph.from_ase(config, cutoff=3.7) for config in test_configs]
        test_graph_batch = to_batch(test_graphs)
        graph_pes_true_energies = test_graph_batch.properties['energy'].detach().numpy()
        graph_pes_true_forces = test_graph_batch.properties['forces'].detach().numpy()
        model = go_mace_23(precision=None)
        graph_pes_pred_energies = model.predict_energy(test_graph_batch).detach().numpy()
        graph_pes_pred_forces = model.predict_forces(test_graph_batch).detach().numpy()

        # now compare our package
        model_name = "GO-MACE-23"
        calculator = load_model_calculator(model_name, device="cuda:0")
        my_true_energies, my_true_forces, my_pred_energies, my_pred_forces, config_lens = [], [], [], [], []
        for config in test_configs:
            my_true_energies.append(config.get_potential_energy())
            my_true_forces.extend(config.get_forces().tolist())
            config.calc = calculator
            my_pred_energies.append(config.get_potential_energy())
            my_pred_forces.extend(config.get_forces().tolist())
            config_lens.append(len(config))

        my_true_energies = np.array(my_true_energies)
        my_true_forces = np.array(my_true_forces)
        my_pred_energies = np.array(my_pred_energies)
        my_pred_forces = np.array(my_pred_forces)

        # also compare the standard method loading of energies and forces to see if its a method error
        all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data([model_name],
                                                                                                                  [test_set_fpath],
                                                                                                                  perform_mean_energy_shift=False,
                                                                                                                  pbc=False)

        print("Graph PES True Energies (eV): ", graph_pes_true_energies[0:10])
        print("My True Energies (eV): ", my_true_energies[0:10])
        print("get_test_and_pred_data Energies (meV / atom): ", all_test_energies[0][0:10])

        print("Graph PES True Forces (eV): ", graph_pes_true_forces[0:10])
        print("My True Forces (eV): ", my_true_forces[0:10])
        print("get_test_and_pred_data Forces (eV): ", all_test_forces[0][0:10])

        print("Graph PES Pred Energies (eV): ", graph_pes_pred_energies[0:10])
        print("My Pred Energies (eV): ", my_pred_energies[0:10])
        print("get_test_and_pred_data Energies (meV / atom): ", all_pred_energies_dict[model_name][0][0:10])

        print("Graph PES Pred Forces: ", graph_pes_pred_forces[0:10])
        print("My Pred Forces: ", my_pred_forces[0:10])
        print("get_test_and_pred_data Forces (meV / A): ", all_pred_forces_dict[model_name][0][0:10])

        # Identify non-close elements
        energy_mask = ~np.isclose(graph_pes_pred_energies, my_pred_energies, rtol=1e-5, atol=1e-8)
        forces_mask = ~np.isclose(graph_pes_pred_forces, my_pred_forces, rtol=1e-4, atol=1e-4)

        print("Graph PES Non-Close Energies: ", graph_pes_pred_energies[energy_mask])
        print("My Code Non-Close Energies: ", my_pred_energies[energy_mask])
        print("Graph PES Non-Close Forces: ", graph_pes_pred_forces[forces_mask])
        print("My Code Non-Close Forces: ", my_pred_forces[forces_mask])

        assert ~np.any(energy_mask)
        assert ~np.any(forces_mask)

        # Now calculate energy RMSES
        ...

        # Now calculate force RMSES
        print("Graph PES Pred Forces: ", graph_pes_pred_forces)
        print("Graph PES True Forces: ", graph_pes_true_forces)
        # print("My Pred Forces Shape: ", my_pred_forces.shape)
        # print("My True Forces Shape: ", my_true_forces.shape)
        print("Graph PES Pred Forces - True Forces: ", graph_pes_pred_forces - graph_pes_true_forces)
        # print("s")

        # vector distance force rmses
        graph_pes_force_rmse = calculate_comp_rmse(graph_pes_pred_forces, graph_pes_true_forces)
        my_force_rmse = calculate_comp_rmse(my_pred_forces, my_true_forces)
        all_pred_forces_rmse = calculate_comp_rmse(all_pred_forces_dict[model_name][0], all_test_forces[0])

        print("Graph PES Comp Force RMSE eV: ", graph_pes_force_rmse)
        print("My Force Comp RMSE eV: ", my_force_rmse)
        print("All Pred Comp Forces RMSE eV: ", all_pred_forces_rmse)

        # force norm rmses
        graph_pes_pred_force_norms = np.linalg.norm(graph_pes_pred_forces, axis=1)
        graph_pes_true_force_norms = np.linalg.norm(graph_pes_true_forces, axis=1)
        graph_pes_force_norm_rmse = calculate_rmse(graph_pes_pred_force_norms, graph_pes_true_force_norms)
        print("Graph PES Force Norm RMSE: ", graph_pes_force_norm_rmse)

        parity_plot(model, structures, property = "forces", transform =  None)


class ModelTests(unittest.TestCase):

    def test_simple_pbc(self):
        """
        Test the energy and force errors for an equivalent system with PBC or no-PBC.
        """

        return
    
    def test_cuequivariance(self):
        """
        Test to confirm whether cuequivariance is working and producing a speedup in inference, dynamics etc.
        """

        import time
        model_names = ['MACE-MP0-L1']
        test_set_fpaths = [f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz"]

        print('Running inference on test set without cuequivariance...')
        start_time = time.time()
        test_energies, test_forces, pred_energies, pred_forces = get_test_and_pred_data(model_names,
                                                                                        test_set_fpaths, 
                                                                                        enable_cueq=False)
        end_time = time.time()
        time_taken_1 = end_time - start_time
        print(f"Counted Configs in {time_taken_1: .4f}s ({time_taken_1 / float(len(test_energies)):.4f}s per config) \
              for {len(test_energies)} configs...")

        print('Running inference on test set with cuequivariance...')
        start_time = time.time()
        test_energies, test_forces, pred_energies, pred_forces = get_test_and_pred_data(model_names,
                                                                                        test_set_fpaths, 
                                                                                        enable_cueq=True)
        end_time = time.time()
        time_taken_2 = end_time - start_time
        print(f"Counted Configs in {time_taken_2: .4f}s ({time_taken_2 / float(len(test_energies)):.4f}s per config) \
              for {len(test_energies)} configs...")

        assert time_taken_2 < time_taken_1


class DescriptorTests(unittest.TestCase):

    def test_soap_desc(self):

        configs_fp = f"{TEST_DATA_DIR}/structures/endo_ace_wB97M_3_config.xyz"
        configs = flexible_ase_read(configs_fp)

        # get all soap dec
        soap_desc = get_soap_descriptors(configs)

        print()


    def test_best_cos_sim(self):

        from ase.data import atomic_numbers as element_to_atomic_num_dict

        # single element case

        array_1 = np.array([np.array([1, 0, 0]), 
                            np.array([1, 0, 0])])
        
        array_2 = np.array([np.array([0, 1, 0]), 
                            np.array([-1, 0, 0])])
        
        elements = ['C', 'C']
        atom_nums = np.array([element_to_atomic_num_dict[element] for element in elements])
        best_sim_tuples, best_sim_mean, best_sim_element_means = calc_best_similarities(array_1, 
                                                                                             array_2, 
                                                                                             atom_nums)

        print('Best Sim Tuples: \n', best_sim_tuples)
        print('Element Means: ', best_sim_element_means)
        print('Overall Mean: ', best_sim_mean)

        assert best_sim_mean == 0.0

        # two element case
        array_1 = np.array([np.array([1, 0, 0]), 
                            np.array([0, 1, 0]), 
                            np.array([1, 0, 0])])
        
        array_2 = np.array([np.array([1, 0, 0]), 
                            np.array([-1, -1, -1]), 
                            np.array([-1, 0, 0])])
        
        elements = ['C', 'C', 'H']
        atom_nums = np.array([element_to_atomic_num_dict[element] for element in elements])
        best_sim_tuples, best_sim_mean, best_sim_element_means = calc_best_similarities(array_1, 
                                                                                             array_2, 
                                                                                             atom_nums, 
                                                                                             element_stratify_mean=True)

        expected_element_means = [1.0, 0.5]

        print('Best Sim Tuples: \n', best_sim_tuples)
        print('Element Means: ', best_sim_element_means)
        print('Overall Mean: ', best_sim_mean)

        assert best_sim_mean == 0.75


class IOTests(unittest.TestCase):


    def test_sample_dataset(self):

        configs_fp = f"{TEST_DATA_DIR}/structures/multi_element.xyz"

        configs = sample_dataset(configs_fp, n=1, element_filter_set={'C'})
        print(configs)
        assert len(configs) == 1

        configs = sample_dataset(configs_fp, n=None, element_filter_set={'H'})
        print(configs)
        assert len(configs) == 2

        configs = sample_dataset(configs_fp, 2, element_filter_set={'O'})
        print(configs)
        assert len(configs) == 2

    def test_npz_cacheing(self):

        npz_cache_fp = f"{TEST_DIR}/out/npz/test_cache.npz"

        energies = np.array([0.0, 1.0, 2.0])
        forces = np.array([np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
        metadata = {'foo': 'bar'}

        save_npz_data_cache(npz_cache_fp, {'energies': energies, 'forces': forces}, meta=metadata)

        cache_data = load_data_cache(npz_cache_fp)
        loaded_energies, loaded_forces, loaded_metadata = cache_data['energies'], cache_data['forces'], cache_data['meta']

        print("Cache Data: ", cache_data)
        print("Loaded Energies: ", loaded_energies)
        print("Loaded Forces: ", loaded_forces)
        print("Cache meta: ", metadata)
        assert np.isclose(energies, loaded_energies).all()
        assert np.isclose(forces, loaded_forces).all()
        assert metadata == loaded_metadata

    def test_npz_cache_vs_pred(self):

        # load xyz e.g. xyz file for predictions
        input_name = "endo_ace_wB97M_3_config"
        input_fp = f"{TEST_DIR}/data/structures/{input_name}.xyz"

        # remove npz caches
        npz_fp = f"{TEST_DIR}/out/npz/{input_name}.npz"
        if os.path.exists(npz_fp):
            os.remove(npz_fp)

        # make predictions with 3 models
        model_names = ['GO-MACE-23', 'MACE-OFF23', 'MACE-MP0-128-L1']
        model_pred_energies, model_pred_forces = get_model_predictions(model_name,
                                                                       test_set_fpaths,
                                                                       include_forces=include_forces,
                                                                       offset_ref_fpath=offset_ref_fpath,
                                                                       pred_isolated_atom_fpath=pred_isolated_atom_fpath,
                                                                       custom_xyz_parsers=custom_xyz_parsers,
                                                                       cuda_device=cuda_device,
                                                                       pbc_box=pbc_box,
                                                                       energy_units=energy_units,
                                                                       force_units=force_units,
                                                                       model_loader=model_loader,
                                                                       enable_cueq=enable_cueq,
                                                                       hdf5_file=hdf5_file)

        # make predictions again (this time it will load from cache)

        # assert that first predictions == second predictions

        npz_cache_fp = f"{TEST_DIR}/out/npz/test_cache.npz"

        energies = np.array([0.0, 1.0, 2.0])
        forces = np.array([np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
        metadata = {'foo': 'bar'}

        save_npz_data_cache(npz_cache_fp, {'energies': energies, 'forces': forces}, meta=metadata)

        cache_data = load_data_cache(npz_cache_fp)
        loaded_energies, loaded_forces, loaded_metadata = cache_data['energies'], cache_data['forces'], cache_data['meta']

        print("Cache Data: ", cache_data)
        print("Loaded Energies: ", loaded_energies)
        print("Loaded Forces: ", loaded_forces)
        print("Cache meta: ", metadata)
        assert np.isclose(energies, loaded_energies).all()
        assert np.isclose(forces, loaded_forces).all()
        assert metadata == loaded_metadata

    def test_ds_sort_cv(self):

        # atom distances
        configs = flexible_ase_read(f"{TEST_DATA_DIR}/structures/atom_distances.xyz")

        # assign ids
        for i, config in enumerate(configs):
            config.info['id'] = i

        # order based on distance of both pairs
        expected_order= [1, 2, 0]
        cv = [(0, 2), (1, 3)]
        sorted_configs = sort_configs_by_cv_dist(configs, cv)
        sort_order = [config.info['id'] for config in sorted_configs]
        assert sort_order == expected_order

        # order only based on distance of one pair
        expected_order= [1, 2, 0]
        cv = [(0, 2)]
        sorted_configs = sort_configs_by_cv_dist(configs, cv)
        sort_order = [config.info['id'] for config in sorted_configs]
        assert sort_order == expected_order


    def test_ds_sort_energies(self):

        # atom distances
        configs = flexible_ase_read(f"{TEST_DATA_DIR}/structures/atom_distances.xyz")
        
        return
    

    def test_ds_sort_soap(self):

        # using spice as an e.g., spice similar
        configs = flexible_ase_read(f"{TEST_DATA_DIR}/structures/h2o_SOAP_sim.xyz")
        pretrain_configs = flexible_ase_read(f"{TEST_DATA_DIR}/structures/h2o_SOAP_sim_2.xyz")

        # assign ids
        for i, config in enumerate(configs):
            config.info['id'] = i

        expected_order = [2, 1, 0]
        sorted_configs = sort_configs_by_soap_desc(configs, 
                                                   pretrain_configs, 
                                                   mode='sim', 
                                                   reverse=False, 
                                                   compress_dim=None)
        sort_order = [config.info['id'] for config in sorted_configs]
        print("Sort order: ", sort_order)
        self.assertEqual(sort_order, expected_order)

        # reverse
        expected_order = [0, 1, 2]
        sorted_configs = sort_configs_by_soap_desc(configs, 
                                                   pretrain_configs, 
                                                   mode='sim', 
                                                   reverse=True, 
                                                   compress_dim=None)
        sort_order = [config.info['id'] for config in sorted_configs]
        print("Sort order: ", sort_order)
        self.assertEqual(sort_order, expected_order)


    def test_save_force_errors_to_xyz(self):

        # open xyz file

        # run predictions

        # save ext xyz with prediction labels

        # re-open and assert predictions have been read correctly

        self.assertEqual(True, True)

    def test_save_dataset_batches(self):

        dataset_fp = f"{DATASETS_DIR}/go_mace/GO-MACE-23_train.xyz"  # go-mace training set
        out_dir = '/'.join(dataset_fp.split('/')[0:-1])
        ds_name = get_name_from_path(dataset_fp)
        batch_size = 1000

        save_dataset_batches(dataset_fp, batch_size)
        num_batches = 4
        input_num_configs = len(get_xyz_config_offsets(dataset_fp))
        num_batched_configs = 0
        for i in range(num_batches):
            batch_len = len(get_xyz_config_offsets(f"{out_dir}/{ds_name}_batch_{i}.xyz"))
            print(f'Batch {i} Len: ', batch_len)
            num_batched_configs += batch_len

        self.assertEqual(input_num_configs, num_batched_configs)


if __name__ == "__main__":

    unittest.main(warnings='ignore', verbosity=2)