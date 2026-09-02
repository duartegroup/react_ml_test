from react_ml_test_public.helpers.settings import DFT_DIR
from react_ml_test_public.helpers.io import check_orca_outputs

if __name__ == "__main__":

    # === INPUT WORK DIR + RUN N AME ===
    work_dir = f"{DFT_DIR}/sp"
    # input_name = "full_al_traj_samples_dist_filtered"
    # input_name = "guajadial_extra_traj_samples_dist_filtered_solv"
    # input_name = "CPD_MVK_DA_test_PBE_def2_TZVP"
    # input_name = "CPD_MVK_DA_test_wB97M_TZVPPD"
    # input_name = "anthracene_DA_test_wB97M_TZVPP"
    # input_name = "guajadial_DA_test_wB97M_TZVPP"
    input_name = "guajadial_DA_test_wB97M_TZVPPD"
    # input_name = "anthracene_DA_test_PBE_def2_TZVP"
    # input_name = "guajadial_DA_test_PBE_def2_TZVP"

    # input_name = "guajadial_extra_traj_samples_dist_filtered_vac"
    # ==================================

    print(f"Checking outputs for {input_name}...")
    check_orca_outputs(work_dir, input_name)
