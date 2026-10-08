"""
Unit tests for the flexible model-colour lookup, ``settings.get_model_colour``.

Plot lines are coloured by looking each model name up in ``PAPER_MODEL_COLOURS``, matching a
key as a stub of the name rather than requiring an exact entry. That is what lets a fine-tuned
/ from-scratch run be plotted in its base model's colour: the names the workflow YAMLs build,
e.g. 'MACE-OFF23_medium_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2', carry
their base model's name as a stub.

These tests use names from the shipped input YAMLs (inputs/irc/*.yaml), so a change to the
naming convention those scripts build is caught here. No MLIP stack is needed.

Run from the repository root:

    pytest testing/test_model_colours.py
"""

import pytest
from matplotlib.colors import to_hex, to_rgb
from matplotlib import pyplot as plt

from react_ml_test_public.helpers import settings
from react_ml_test_public.helpers.settings import PAPER_MODEL_COLOURS, get_model_colour

# the fallback colours are drawn from tab10 (matplotlib's default colour cycle)
TAB10_TO_HEX = {f"C{index}": to_hex(plt.cm.tab10(index)) for index in range(10)}
TAB10_COLOURS = list(TAB10_TO_HEX.values())
# the colours PAPER_MODEL_COLOURS already uses, as hex (so 'grey' compares as '#808080')
MAPPED_COLOURS = {to_hex(colour) for colour in PAPER_MODEL_COLOURS.values()}

# base models and the fine-tuned names the reaction-barrier / test-set YAMLs build from them,
# as resolved by inputs.resolve_reaction_barrier_comparison_yaml for inputs/irc/*.yaml
BASE_AND_FINE_TUNED_NAMES = [
    ("GO-MACE-23", "GO-MACE-23_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2"),
    ("MACE-OFF23_medium", "MACE-OFF23_medium_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2"),
    ("MACE-MP-0b3_medium", "MACE-MP-0b3_medium_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2"),
    ("MACE-omol-0-extra-large-1024",
     "MACE-omol-0-extra-large-1024_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_3"),
]


class TestGetModelColour:

    @pytest.mark.parametrize("model_name", list(PAPER_MODEL_COLOURS))
    def test_exact_name_matches(self, model_name):
        assert get_model_colour(model_name) == PAPER_MODEL_COLOURS[model_name]

    @pytest.mark.parametrize("base_model_name, fine_tuned_name", BASE_AND_FINE_TUNED_NAMES)
    def test_fine_tuned_name_inherits_base_model_colour(self, base_model_name, fine_tuned_name):
        assert get_model_colour(fine_tuned_name) == PAPER_MODEL_COLOURS[base_model_name]

    def test_stub_matches_anywhere_in_the_name(self):
        """The stub need not be a prefix: the fine-tuning components come before it too."""
        assert get_model_colour("from_scratch_runs_EGRET_1T_v2") == PAPER_MODEL_COLOURS["EGRET_1T"]

    def test_different_models_get_different_colours(self):
        """A colour per model, not one colour per position in a list."""
        colours = [get_model_colour(name) for _, name in BASE_AND_FINE_TUNED_NAMES]
        assert len(set(colours)) == len(colours)

    def test_unmatched_name_gets_a_tab10_colour(self):
        """A model the map does not cover is still given a colour, rather than grey or None."""
        colour = get_model_colour("NOT-A-REAL-MODEL")
        assert colour in TAB10_COLOURS
        assert colour not in MAPPED_COLOURS

    def test_unmatched_name_gets_no_tab10_colour_the_map_uses(self):
        """
        The fallback must not collide with a paper model's colour - neither an exact repeat
        (the tab10 blue MACE-OFF23_medium uses) nor a near one (tab10's grey against 'REF').
        """
        unmatched_names = ["NOT-A-REAL-MODEL", "uma-s-1p2p1", "esen-sm-filtered-odac25",
                           "some-other-model", "MACE-ANI-CC", ""]
        fallbacks = {name: get_model_colour(name) for name in unmatched_names}
        assert all(colour not in MAPPED_COLOURS for colour in fallbacks.values())
        assert TAB10_TO_HEX["C0"] not in fallbacks.values()  # MACE-OFF23_medium's blue
        assert TAB10_TO_HEX["C7"] not in fallbacks.values()  # 'REF' grey

    def test_fallback_is_stable_for_a_name(self):
        """The same model keeps its colour in every figure and every run."""
        assert get_model_colour("NOT-A-REAL-MODEL") == get_model_colour("NOT-A-REAL-MODEL")

    def test_fallback_is_not_the_same_for_every_name(self):
        colours = {get_model_colour(name) for name in
                   ["NOT-A-REAL-MODEL", "uma-s-1p2p1", "esen-sm-filtered-odac25", "other-model"]}
        assert len(colours) > 1

    def test_explicit_default_still_wins(self):
        assert get_model_colour("NOT-A-REAL-MODEL", default="grey") == "grey"

    def test_none_default_reports_the_unmatched_case(self):
        """The input validation relies on None meaning 'no colour mapped'."""
        assert get_model_colour("NOT-A-REAL-MODEL", default=None) is None
        assert get_model_colour("", default=None) is None
        assert get_model_colour("GO-MACE-23", default=None) == PAPER_MODEL_COLOURS["GO-MACE-23"]

    def test_fallback_avoids_the_colours_of_a_replaced_map(self, monkeypatch):
        """The fallback is drawn from the current map, so editing the map moves it too."""
        monkeypatch.setattr(settings, "PAPER_MODEL_COLOURS",
                            {"A": TAB10_TO_HEX["C1"], "B": TAB10_TO_HEX["C2"]})
        colour = get_model_colour("unmapped-model")
        assert colour in TAB10_COLOURS
        assert colour not in {TAB10_TO_HEX["C1"], TAB10_TO_HEX["C2"]}

    def test_longest_stub_wins_regardless_of_map_order(self, monkeypatch):
        """
        A name matching several stubs takes the longest, so the colour does not depend on the
        order PAPER_MODEL_COLOURS happens to be written in.
        """
        colour_map = {"MACE": "red", "MACE-OFF23_medium": "blue"}
        monkeypatch.setattr(settings, "PAPER_MODEL_COLOURS", colour_map)

        assert get_model_colour("MACE-OFF23_medium_fine_tuned") == "blue"
        assert get_model_colour("MACE-OFF23_small") == "red"

    def test_reversed_map_order_gives_the_same_colour(self, monkeypatch):
        monkeypatch.setattr(settings, "PAPER_MODEL_COLOURS",
                            {"MACE-OFF23_medium": "blue", "MACE": "red"})
        assert get_model_colour("MACE-OFF23_medium_fine_tuned") == "blue"


class TestShippedConfigsAreColoured:
    """Every model the shipped reaction-barrier YAMLs list must resolve to a colour."""

    @pytest.mark.parametrize("yaml_fpath", ["inputs/irc/anthracene_1_4_DA_IRC_barrier.yaml",
                                            "inputs/irc/anthracene_DA_IRC_barrier.yaml",
                                            "inputs/irc/CPD_MVK_DA_IRC_barrier.yaml",
                                            "inputs/irc/guajadial_DA_IRC_barrier.yaml"])
    def test_model_names_all_resolve(self, yaml_fpath):
        from react_ml_test_public.helpers.inputs import resolve_reaction_barrier_comparison_yaml

        model_names = resolve_reaction_barrier_comparison_yaml(yaml_fpath)["model_names"]
        assert model_names
        missing = [name for name in model_names if get_model_colour(name, default=None) is None]
        assert not missing, f"{yaml_fpath} lists models with no PAPER_MODEL_COLOURS stub: {missing}"

    def test_fine_tuned_models_keep_distinct_base_colours(self):
        from react_ml_test_public.helpers.inputs import resolve_reaction_barrier_comparison_yaml

        model_names = resolve_reaction_barrier_comparison_yaml(
            "inputs/irc/anthracene_1_4_DA_IRC_barrier.yaml")["model_names"]
        # the four base models of the paper, each distinctly coloured
        assert len({get_model_colour(name) for name in model_names}) == 4


class TestTestSetComparisonColours:
    """The colour list run_test_set_comparison.py hands the plot helpers is one per model."""

    @pytest.fixture(scope="class")
    def model_names(self):
        from react_ml_test_public.helpers.inputs import resolve_test_set_comparison_yaml

        return resolve_test_set_comparison_yaml(
            "inputs/test/CPD_MVK_DA_FT42_test.yaml")["model_names"]

    def test_one_colour_per_model(self, model_names):
        colours = [get_model_colour(name, default=None) for name in model_names]
        assert len(colours) == len(model_names)
        assert all(colour is not None for colour in colours)

    def test_base_model_and_its_fine_tuned_run_share_a_colour(self, model_names):
        """
        The YAML alternates each base model with its fine-tuned run, so the base name is a
        stub of the fine-tuned name and the pair must be plotted in the same colour.
        """
        for base_name, fine_tuned_name in zip(model_names[::2], model_names[1::2]):
            assert base_name in fine_tuned_name
            assert get_model_colour(fine_tuned_name) == get_model_colour(base_name)

    def test_fine_tuned_runs_are_no_longer_coloured_by_position(self, model_names):
        """
        Reordering the models must not change any model's colour, which is what the old
        positional colour list got wrong.
        """
        colours = {name: get_model_colour(name) for name in model_names}
        for name in reversed(model_names):
            assert get_model_colour(name) == colours[name]
