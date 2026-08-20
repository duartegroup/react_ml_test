"""
Script to save all ORCA outputs in a given directory with a specified input name into a single .npz + extended .xyz file.
"""
import sys
sys.path.append('../')
import glob
import os

from mlptrain.utils import orca_output_to_npz

from react_ml_test.helpers.settings import DFT_DIR

if __name__ == "__main__":

    # specify working directory
    work_dir = f"{DFT_DIR}/sp"
    # input_name = "full_al_traj_samples_dist_filtered"
    # input_name = "guajadial_extra_traj_samples_dist_filtered_solv"
    # input_name = "anthracene_1_4_MMFF_vac_MACE-OFF23_46_reaction"
    # input_name = "anthracene_1_4_DA_vac_OFF46_opt_TS"
    # input_name = "anthracene_1_4_DA_in_toluene_OFF46_opt_TS_cutoff=5_2"
    # input_name = "CPD_MVK_DA_test_PBE"
    # input_name = "CPD_MVK_DA_test_PBE_def2_TZVP"
    # input_name = "CPD_MVK_DA_test_wB97M_TZVPPD"
    # input_name = "anthracene_DA_test_wB97M_TZVPP"
    # input_name = "anthracene_DA_test_wB97M_TZVPPD"
    # input_name = "guajadial_DA_test_wB97M_TZVPP"
    input_name = "guajadial_DA_test_wB97M_TZVPPD"
    # input_name = "anthracene_DA_test_PBE_def2_TZVP"
    # input_name = "guajadial_DA_test_PBE_def2_TZVP"
    # input_name = "endo_DA_wB97M_combined_test_PBE"
    # ===========================

    # get out_fpaths
    out_fpaths = glob.glob(os.path.join(work_dir, f"**/*{input_name}_*.out"), recursive=True)

    orca_output_to_npz(out_fpaths, input_name, work_dir)