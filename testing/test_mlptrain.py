"""
External tests for mlptrain.
"""
import os
import unittest
import sys
import torch
import mlptrain as mlt
from ase.io.trajectory import Trajectory as ASETrajectory
from .data.utils import work_in_zipped_dir
from autode.atoms import Atom

here = os.path.abspath(os.path.dirname(__file__))

def h2o():
    """Water molecule"""
    atoms = [
        Atom('H', 2.32670, 0.51322, 0.0),
        Atom('H', 1.03337, 0.70894, -0.89333),
        Atom('O', 1.35670, 0.51322, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)

class MLPTrainTests(unittest.TestCase):

    @work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
    def test_mace_gpu_cueq(self):
        """Test MACE MD run with gpu + cuequivariance acceleration. """

        mlt.Config.n_cores = 4
        mlt.Config.mace_device = 'cuda'
        mlt.Config.mace_params['calc_device'] = 'cuda'
        mlt.Config.mace_params['cueq'] = True
        assert torch.cuda.is_available()

        # Check the default device
        print("Torch Version: ", torch.__version__)
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA Version: {torch.version.cuda}")

        # H2O molecule
        system = mlt.System(h2o(), box=[50, 50, 50])
        config = system.random_configuration()

        # create MACE potential
        mace = mlt.potentials.MACE('water', system=system, foundation='medium_off')

        # run some dynamics with the potential
        mlt.md.run_mlp_md(
                configuration=config,
                mlp=mace,
                temp=300,
                dt=1,
                interval=10,
                kept_substrings=['.traj'],
                ps=0.5
        )

        assert os.path.exists('trajectory.traj')
        traj = ASETrajectory('trajectory.traj')

        # 500 fs / 10 + 1 starting frame
        assert len(traj) == 50 + 1
