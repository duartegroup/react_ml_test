"""
Script to save a sample of an xyz dataset very efficiently, without opening the entire dataset.
"""
import os
import time

from ase.io import write

import glob

from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test_public.helpers.settings import TRAJ_DIR
from react_ml_test_public.helpers.utils import natural_sort_key

if __name__ == "__main__":

    # === SPECIFY INPUT ===

    work_dir = TRAJ_DIR
    # traj_name_stub = "CPD_MVK_DA_water*ts_gauss"
    traj_name_stub = "anthracene*toluene"
    # traj_name_stub = "guajadial*water*"
    frame_indices = (-1, -1)

    # =====================================

    fpaths = sorted(glob.glob(os.path.join(work_dir, f"{traj_name_stub}*.xyz")), key=natural_sort_key)
    print(f"Found {len(fpaths)} trajectories for name stub: {traj_name_stub}...")
    assert len(fpaths) > 0, f"No files detected in directory {traj_dir} with name stub {traj_name_stub}..."        
    
    for fpath in fpaths:

        if 'frames' in fpath:
            continue
        
        if frame_indices == (-1, -1):
            traj_frames = flexible_ase_read(fpath)[frame_indices[0]:]
        else:
            traj_frames = flexible_ase_read(fpath)[frame_indices[0]:frame_indices[1]]
        
        out_fp = f"{work_dir}/{get_name_from_path(fpath)}_frames.xyz"
        write(out_fp, traj_frames, format='extxyz')
        print(f"Saved frames to {out_fp}...")
