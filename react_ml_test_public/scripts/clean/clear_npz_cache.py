"""
Remove all cached .npz files in out/npz/ (regenerated on demand by other
rmlt workflows).

Run from the repo root: `rmlt clear_npz_cache`
"""

import glob
import os

from react_ml_test_public.helpers.settings import NPZ_DIR

if __name__ == "__main__":

    npz_files = sorted(glob.glob(f"{NPZ_DIR}/*.npz"))
    n_files = len(npz_files)

    if n_files == 0:
        print(f"No .npz files found in {NPZ_DIR}.")
        raise SystemExit

    prompt = (f"Are you sure you want to remove {n_files} .npz file"
              f"{'s' if n_files != 1 else ''} in {NPZ_DIR}? (y/n): ")
    response = input(prompt).strip().lower()

    if response not in ("y", "yes"):
        print("Aborting - no files removed.")
        raise SystemExit

    for npz_fpath in npz_files:
        os.remove(npz_fpath)

    print(f"Removed {n_files} .npz file{'s' if n_files != 1 else ''} from {NPZ_DIR}.")
