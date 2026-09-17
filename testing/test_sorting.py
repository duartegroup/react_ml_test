"""
Unit tests for the dataset sorting helpers in ``react_ml_test_public.helpers.sorting``.

These use in-memory ``ase.Atoms`` objects (no MLIP stack needed), giving each config an
explicit force array so the expected order is known exactly.

Run from the repository root:

    pytest testing/test_sorting.py
"""

import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule

from react_ml_test_public.helpers.io import get_ase_atoms_forces
from react_ml_test_public.helpers.sorting import sort_configs_by_forces


def config_with_forces(forces, symbol: str = "H") -> Atoms:
    """A config of one atom per force vector, carrying those forces."""
    forces = np.asarray(forces, dtype=float)
    config = Atoms(symbol * len(forces), positions=np.zeros((len(forces), 3)))
    config.arrays["forces"] = forces
    return config


def summed_force_norm(config: Atoms) -> float:
    """The quantity sort_configs_by_forces orders by: sum of the atomic force norms."""
    return float(np.linalg.norm(get_ase_atoms_forces(config), axis=1).sum())


class TestSortConfigsByForces:

    def test_sorts_low_to_high_by_default(self):
        configs = [config_with_forces([[5.0, 0.0, 0.0]] * 3),
                   config_with_forces([[1.0, 0.0, 0.0]] * 3),
                   config_with_forces([[3.0, 0.0, 0.0]] * 3)]

        sorted_configs = sort_configs_by_forces(configs)

        assert [summed_force_norm(config) for config in sorted_configs] == [3.0, 9.0, 15.0]

    def test_reverse_sorts_high_to_low(self):
        configs = [config_with_forces([[5.0, 0.0, 0.0]] * 3),
                   config_with_forces([[1.0, 0.0, 0.0]] * 3),
                   config_with_forces([[3.0, 0.0, 0.0]] * 3)]

        sorted_configs = sort_configs_by_forces(configs, reverse=True)

        assert [summed_force_norm(config) for config in sorted_configs] == [15.0, 9.0, 3.0]

    def test_orders_by_vector_norm_not_component_sum(self):
        """(6, 0, 0) has a larger norm but a smaller component sum than (4, 4, 0)."""
        larger_norm = config_with_forces([[6.0, 0.0, 0.0]])
        smaller_norm = config_with_forces([[4.0, 4.0, 0.0]])

        assert summed_force_norm(smaller_norm) < summed_force_norm(larger_norm)
        assert abs(smaller_norm.arrays["forces"]).sum() > abs(larger_norm.arrays["forces"]).sum()

        sorted_configs = sort_configs_by_forces([larger_norm, smaller_norm])

        assert sorted_configs == [smaller_norm, larger_norm]

    def test_input_list_is_not_modified(self):
        configs = [config_with_forces([[5.0, 0.0, 0.0]] * 3),
                   config_with_forces([[1.0, 0.0, 0.0]] * 3)]

        sorted_configs = sort_configs_by_forces(configs)

        assert sorted_configs is not configs
        assert [summed_force_norm(config) for config in configs] == [15.0, 3.0]
        assert sorted(id(config) for config in sorted_configs) == sorted(id(config) for config in configs)

    def test_empty_input(self):
        assert sort_configs_by_forces([]) == []

    def test_missing_forces_raises_and_names_the_config(self):
        configs = [config_with_forces([[1.0, 0.0, 0.0]] * 3), molecule("CH4")]

        with pytest.raises(RuntimeError, match="config 1 of 2"):
            sort_configs_by_forces(configs)

    def test_missing_forces_raises_without_modifying_the_input(self):
        configs = [config_with_forces([[5.0, 0.0, 0.0]] * 3),
                   config_with_forces([[1.0, 0.0, 0.0]] * 3),
                   molecule("CH4")]
        order_before = [id(config) for config in configs]

        with pytest.raises(RuntimeError):
            sort_configs_by_forces(configs)

        assert [id(config) for config in configs] == order_before

    def test_alternative_force_array_labels(self):
        """Forces written as QM_forces / REF_forces in an extxyz file are accepted too."""
        qm_forces = config_with_forces([[4.0, 0.0, 0.0]] * 3)
        ref_forces = config_with_forces([[2.0, 0.0, 0.0]] * 3)
        qm_forces.arrays["QM_forces"] = qm_forces.arrays.pop("forces")
        ref_forces.arrays["REF_forces"] = ref_forces.arrays.pop("forces")

        sorted_configs = sort_configs_by_forces([qm_forces, ref_forces])

        assert sorted_configs == [ref_forces, qm_forces]
        assert [summed_force_norm(config) for config in sorted_configs] == [6.0, 12.0]

    def test_forces_from_an_attached_calculator(self):
        """Forces need not be pre-loaded: a calculator result is used when present."""
        emt = pytest.importorskip("ase.calculators.emt")
        configs = []
        for name in ("CH4", "NH3"):
            config = molecule(name)
            config.calc = emt.EMT()
            configs.append(config)

        sorted_configs = sort_configs_by_forces(configs)

        norms = [summed_force_norm(config) for config in sorted_configs]
        assert norms == sorted(norms)
