"""
Config file containing file paths for data and outputs, and other general settings.
"""
import os
import platform
import random
from string import Template
from typing import List, Literal, NamedTuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb

#  os.environ.get("USER")

# set root dir for inputs / outputs
if any(x in platform.node() for x in ('htc', 'arc')):
    root_data_dir = os.path.join(os.environ["DATA"] + '/')
else:
    root_data_dir = "./" 

# input paths
MODEL_DIR = f"{root_data_dir}models" if any(x in platform.node() for x in ('htc', 'arc')) else "models"  
MODELS = ["GO-MACE-23",
          "MACE-OFF23_medium",
          "MACE-MP-0b3-medium",
          "ani-1xnr",
          "EGRET_1T",
          "MACE-ANI-CC",
          "MACE-NEB",
          "MACE-omol-0-extra-large-1024"
          ]

MACE_MODELS = ["GO-MACE-23",
               "MACE-OFF23_medium",
               "MACE-MP-0b3-medium",
               "MACE-ANI-CC"]

PAPER_MODELS = ["GO-MACE-23",
                "MACE-OFF23_medium",
                "MACE-MP-0b3_medium"  # updated mp-0 model
                ]

PAPER_MODEL_COLOURS = {"GO-MACE-23": '#B82373',
                       "MACE-OFF23_medium": '#1f77b4',
                       "MACE-MP-0b3_medium": '#22AB54',
                       "MACE-omol-0-extra-large-1024": "#ffa600",
                       "MACE-MP0-128-L1": "#22AB54",
                       "EGRET_1T": "red",
                       "ani-1xnr": "pink",
                       "MACE-NEB": "purple",
                       "REF": "grey"
                       }

# sentinel for get_model_colour's ``default``: fall back to a random tab10 colour (so a fixed
# colour is asked for by name, and ``default=None`` still means 'return None when unmatched')
_RANDOM_TAB10_COLOUR = object()

# how close (in RGB) a tab10 colour may come to one already in PAPER_MODEL_COLOURS before it
# counts as the same colour in the fallback above
_MIN_FALLBACK_COLOUR_DISTANCE = 0.05


def get_model_colour(model_name: str,
                     default: object = _RANDOM_TAB10_COLOUR) -> Optional[str]:
    """
    Colour a model is plotted in, looked up flexibly from PAPER_MODEL_COLOURS.

    A model takes a colour when one of the PAPER_MODEL_COLOURS keys is a stub of its name, so a
    fine-tuned / from-scratch run (e.g. 'MACE-OFF23_medium_<train_name>_<n>_config_fine_tuned')
    is coloured as its base model without needing an entry of its own. The longest matching stub
    wins, so a name matching several stubs is coloured the same way whatever order the colour map
    is written in.

    A model matching no stub gets a random tab10 colour that no PAPER_MODEL_COLOURS entry
    already uses (see _random_model_colour); pass ``default`` to get a fixed colour, or None to
    detect the unmatched case instead of colouring it (as the input validation does).

    :param model_name: model name (or plot label) to look a colour up for
    :param default: colour used when no PAPER_MODEL_COLOURS key is a stub of the name, or None
                    to return None for such a name
    :return: the colour, or ``default`` (None) when the name has no mapped colour
    """
    matching_stubs = [stub for stub in PAPER_MODEL_COLOURS if stub in model_name]
    if matching_stubs:
        return PAPER_MODEL_COLOURS[max(matching_stubs, key=len)]
    if default is not _RANDOM_TAB10_COLOUR:
        return default

    return _random_model_colour(model_name)


def _random_model_colour(model_name: str) -> str:
    """
    A random tab10 colour for a model that has no colour in PAPER_MODEL_COLOURS.

    Drawn from the tab10 colours no PAPER_MODEL_COLOURS entry already uses, seeded by the model
    name so that a model keeps its colour across every figure and every run rather than changing
    from plot to plot. Two unmatched models can still draw the same colour.
    """
    free_colours = _tab10_colours_free_of_paper_colours()
    if not free_colours:
        # every tab10 colour is mapped already, so the exclusion above cannot be honoured
        free_colours = [to_hex(plt.cm.tab10(index)) for index in range(10)]

    return random.Random(model_name).choice(free_colours)


def _tab10_colours_free_of_paper_colours() -> List[str]:
    """The tab10 colours that are not a colour PAPER_MODEL_COLOURS already uses."""

    def rgb_distance(rgb_a: tuple, rgb_b: tuple) -> float:
        return sum((a - b) ** 2 for a, b in zip(rgb_a, rgb_b)) ** 0.5

    paper_rgbs = [to_rgb(colour) for colour in PAPER_MODEL_COLOURS.values()]
    free_colours = []
    for index in range(10):
        colour = to_hex(plt.cm.tab10(index))
        # a small tolerance, so a near repeat counts as a match - tab10's grey (#7f7f7f) against
        # PAPER_MODEL_COLOURS' 'grey' (#808080), say, as well as exact repeats such as the
        # tab10 blue that MACE-OFF23_medium uses
        if all(rgb_distance(to_rgb(colour), paper_rgb) > _MIN_FALLBACK_COLOUR_DISTANCE
               for paper_rgb in paper_rgbs):
            free_colours.append(colour)

    return free_colours


ANI_MODELS = ["ani-1xnr"]

FORMATTED_MODEL_NAMES = {"MACE-OFF23_small": "MACE-OFF-23 Small",
                        "MACE-OFF23_medium": "MACE-OFF-23",
                        "GO-MACE-23": "GO-MACE-23",
                        "MACE-MP0-128-L0": "MACE-MP0 Small",
                        "MACE-MP0-128-L1": "MACE-MP0",
                        "ani500k_large_CC": "MACE-ANI-CC",
                        "ani-1xnr": "ANI-1xnr",
                        "MACE-OFF23_medium_endo_DA_1.0_train_from_scratch":"MACE From Scratch"
                        }

# FairChem (https://fair-chem.github.io) models are fetched by name from their hub rather than
# from a *.model file, so they are recognised by name alone. FairChem and MACE cannot be
# installed in the same env, so a fairchem model is only loadable from the separate fair_chem
# env - see the README. The names are the ones fairchem.core.pretrained_mlip.available_models
# serves; the eSEN / AllScAI entries name the task they are trained for in the name itself.
FAIRCHEM_MODELS = ["uma-s-1p2p1",
                   "uma-s-1p2",
                   "uma-s-1p1",
                   "uma-m-1p1",
                   "esen-md-direct-all-omol",
                   "esen-sm-conserving-all-omol",
                   "esen-sm-direct-all-omol",
                   "allscaip-md-conserving-all-omol",
                   "allscaip-md-direct-all-omol",
                   "esen-sm-conserving-all-oc25",
                   "esen-md-direct-all-oc25",
                   "esen-sm-filtered-odac25",
                   "esen-sm-full-odac25"
                   ]

MACE_MODEL_LOADER = 'MACE'
ANI_MODEL_LOADER = 'ani'
GRAPH_PES_MODEL_LOADER = 'graph-pes'
FAIRCHEM_MODEL_LOADER = 'fairchem'
MODEL_LOADERS = [MACE_MODEL_LOADER, ANI_MODEL_LOADER, GRAPH_PES_MODEL_LOADER,
                 FAIRCHEM_MODEL_LOADER]

# input data directories
DATA_DIR =  f"{root_data_dir}data"

DATASETS_DIR = f"{DATA_DIR}/datasets"
DA_PAPER_DATA_DIR = f"{DATASETS_DIR}/da_paper_data"
MD_INPUT_DIR = f"{DATA_DIR}/md_inputs"
ISOLATED_ATOMS_DIR = f"{DATA_DIR}/isolated_atoms"
SOLVENTS_DIR = f"{DATA_DIR}/solvents"

# replay dataset directories
REPLAY_DATASET_DIR = f"{DATA_DIR}/replay_datasets"

# reaction data dirs
REACTION_DATA_DIR = f"{DATA_DIR}/reaction_data"

CPD_MVK_DA_DATA_DIR = f"{REACTION_DATA_DIR}/CPD_MVK_DA"
ANTHRACENE_DA_DATA_DIR = f"{REACTION_DATA_DIR}/anthracene_DA"
GUAJADIAL_DA_DATA_DIR = f"{REACTION_DATA_DIR}/guajadial_DA"
I_Cl_SN2_DATA_DIR = f"{REACTION_DATA_DIR}/I_Cl_SN2"

ANI_1XNR_MODEL_DIR = "../ani-1xnr/model"
TORCH_ANI_RESOURCES_DIR = "../torchani/resources" 

# output paths
OUT_DIR =  f"{root_data_dir}out" if any(x in platform.node() for x in ('htc', 'arc')) else "out"
OPT_DIR = f"{OUT_DIR}/opt"
TRAJ_DIR = f"{OUT_DIR}/trj"
SAMPLES_DIR = f"{OUT_DIR}/samples"
DFT_DIR = f"{OUT_DIR}/dft"
FINAL_CONFIGS_DIR = f"{TRAJ_DIR}/final_configs"
GEOM_OPT_DIR = f"{OUT_DIR}/geom_opt"
VIB_DIR = f"{OUT_DIR}/vib"
PLOT_DIR = f"{OUT_DIR}/plt"
CSV_DIR = f"{OUT_DIR}/csv"
MD_CHECKPOINT_DIR = f"{OUT_DIR}/md_checkpoints"

# interval between MD checkpoint writes, in ps, when a run does not ask for another one (see
# helpers.dynamics.write_md_checkpoint)
DEFAULT_CHECKPOINT_EVERY_PS = 10
NPZ_DIR = f"{OUT_DIR}/npz"
SAMPLES_OUT_DIR = f"{OUT_DIR}/samples"
DESCRIPTORS_DIR = f"{OUT_DIR}/descriptors"
SIMILARITIES_DIR = f"{OUT_DIR}/similarities"
CHEMISCOPE_DIR = f"{OUT_DIR}/chemiscope"


def expand_env_vars(path: str) -> str:
    """
    Expand ``$VAR`` / ``${VAR}`` references in a path from the environment.

    Lets a YAML input name a location that is only known at run time - e.g. the shared
    ``$DATA/models`` on a cluster, which has no fixed absolute path::

        model_dir: "$DATA/models"

    A variable that is not set is an error rather than being left in place, because a
    literal ``$DATA/models`` directory would otherwise be searched and reported as a
    missing model, hiding the real cause. ``$$`` gives a literal ``$``.

    Note this is separate from, and applied after, the ``${...}`` templating that a YAML
    input does against its own keys (see ``substitute_yaml_variables``), so a
    ``${data_dir}``-style reference to another field is resolved before this runs and is
    never mistaken for an unset environment variable.

    :param path: path that may contain ``$VAR`` / ``${VAR}`` references.
    :return: the path with those references expanded.
    :raises ValueError: if a referenced variable is not set, or the reference is malformed.
    """
    try:
        return Template(str(path)).substitute(os.environ)
    except KeyError as exc:
        name = exc.args[0]
        raise ValueError(
            f"{name!r} in {path!r} is neither an environment variable nor a key defined in "
            f"this input YAML (use '$$' for a literal '$')"
        ) from None
    except ValueError as exc:
        raise ValueError(
            f"could not expand the environment variables in {path!r}: {exc} "
            f"(use '$$' for a literal '$')"
        ) from None


def resolve_out_root(out_dir: str = None) -> str:
    """
    Resolve an output root directory, defaulting to ``OUT_DIR``.

    A YAML input can set ``out_dir`` to write its outputs somewhere other than the default
    ``OUT_DIR`` (e.g. when running from a directory other than the repository root); passing
    None falls back to ``OUT_DIR``. ``$VAR`` / ``${VAR}`` references are expanded from the
    environment (see ``expand_env_vars``).

    :param out_dir: output root directory, or None to use OUT_DIR.
    :return: the output root directory.
    """
    return OUT_DIR if out_dir is None else expand_env_vars(out_dir)


def resolve_out_dirs(out_dir: str = None) -> tuple[str, str, str]:
    """
    Resolve an output root directory into the subdirectories used by the workflow scripts.

    The subdirectory layout matches the one defined above, i.e. plots under
    ``<out_dir>/plt`` and CSVs under ``<out_dir>/csv`` (see ``resolve_out_root`` for how
    ``out_dir`` itself is resolved).

    :param out_dir: output root directory, or None to use OUT_DIR.
    :return: (out_dir, plot_dir, csv_dir)
    """
    out_dir = resolve_out_root(out_dir)
    return out_dir, f"{out_dir}/plt", f"{out_dir}/csv"


def resolve_traj_dir(out_dir: str = None) -> str:
    """
    Resolve the MD trajectory output directory under an output root directory.

    Trajectories go under ``<out_dir>/trj`` - matching the ``TRAJ_DIR`` layout the default
    ``OUT_DIR`` gives - so a YAML that sets ``out_dir`` (e.g. for run_mlip_md.py) writes them
    alongside the rest of that run's outputs (see ``resolve_out_root``).

    :param out_dir: output root directory, or None to use OUT_DIR.
    :return: the trajectory output directory.
    """
    return f"{resolve_out_root(out_dir)}/trj"


def resolve_csv_dir(out_dir: str = None) -> str:
    """
    Resolve the CSV output directory under an output root directory.

    CSVs go under ``<out_dir>/csv`` - matching the ``CSV_DIR`` layout the default ``OUT_DIR``
    gives - so a YAML that sets ``out_dir`` (e.g. for run_mlip_md.py) writes them alongside
    the rest of that run's outputs (see ``resolve_out_root``).

    :param out_dir: output root directory, or None to use OUT_DIR.
    :return: the CSV output directory.
    """
    return f"{resolve_out_root(out_dir)}/csv"


def resolve_md_checkpoint_dir(out_dir: str = None) -> str:
    """
    Resolve the MD checkpoint output directory under an output root directory.

    Checkpoints go under ``<out_dir>/md_checkpoints`` - matching the ``MD_CHECKPOINT_DIR``
    layout the default ``OUT_DIR`` gives - so a YAML that sets ``out_dir`` (e.g. for
    run_mlip_md.py) writes them alongside the rest of that run's outputs (see
    ``resolve_out_root``).

    :param out_dir: output root directory, or None to use OUT_DIR.
    :return: the MD checkpoint output directory.
    """
    return f"{resolve_out_root(out_dir)}/md_checkpoints"


def resolve_data_path(fpath: str, data_dir: str = None) -> str:
    """
    Resolve a YAML input's data path against a data root directory.

    Absolute paths, and paths that already resolve from the current working directory (so
    existing inputs that spell out e.g. ``data/reaction_data/...`` keep working), are returned
    unchanged; anything else is taken to be relative to ``data_dir`` (default ``DATA_DIR``).
    A path that resolves under neither is returned as written, so a typo is reported against the
    path the input actually spells out. Both arguments may reference environment variables
    (see ``expand_env_vars``).

    :param fpath: data file path from the input YAML.
    :param data_dir: data root directory, or None to use DATA_DIR.
    :return: the resolved path.
    """
    if fpath is None:
        return fpath
    fpath = expand_env_vars(fpath)
    if os.path.isabs(fpath) or os.path.exists(fpath):
        return fpath
    if data_dir is not None:
        data_dir = expand_env_vars(data_dir)
    data_fpath = os.path.join(DATA_DIR if data_dir is None else str(data_dir), fpath)
    return data_fpath if os.path.exists(data_fpath) else fpath


def resolve_model_dir(model_dir: str = None) -> str:
    """
    Resolve a model directory, defaulting to ``MODEL_DIR``.

    A YAML input can set ``model_dir`` to load its models from somewhere other than the
    default ``MODEL_DIR`` - e.g. a repository that keeps its own ``models/`` directory
    alongside its own ``inputs/`` (the same way ``data_dir`` and ``out_dir`` let it keep
    its own data and outputs), or a location only known at run time such as the shared
    ``$DATA/models`` on a cluster. Passing None falls back to ``MODEL_DIR``, which is what
    every input that does not set the field gets. ``$VAR`` / ``${VAR}`` references are
    expanded from the environment (see ``expand_env_vars``).

    :param model_dir: model directory, or None to use MODEL_DIR.
    :return: the model directory.
    """
    return MODEL_DIR if model_dir is None else expand_env_vars(model_dir)

# TEST PATHS
TEST_DIR = "testing"
TEST_DATA_DIR = f"{TEST_DIR}/data"
TEST_TRJ_DIR = f"{TEST_DATA_DIR}/trj"

TEST_OUT_DIR = f"{TEST_DIR}/out"
TEST_PLT_DIR = f"{TEST_OUT_DIR}/plt"

# UNITS
PS = 1000
NS = 1000000
eV_to_mEV = 1000
eV_to_kJ_mol = 96.48553482856799
eV_to_kcal_mol = 23.06059627833843
HARTREE_TO_EV = 27.2114079527  # ase.units.Hartree, i.e. 1 Ha in eV

EV = 'eV'
MEV = 'meV'
MEV_PER_ATOM = 'meV / atom'
KJ_PER_MOL = 'kJ / mol'
KCAL_PER_MOL = 'kcal / mol'

EV_PER_AMSTRONG = 'eV / A'
MEV_PER_AMSTRONG = 'meV / A'

STANDARD_CHARGE = 0
STANDARD_MULT = 1
SINGLE_POINT = 'single_point'
GEOM_OPT = 'opt'
TS_OPT = 'ts_opt'

# possible energy and force lables
POSSIBLE_ENERGY_LABELS = ['energy', 'QM_energy', 'E']
POSSIBLE_FORCE_LABELS = ['forces', 'QM_forces']

class MDSettings(NamedTuple):
    """
    Settings struct for an md run.
    """
    fs: int
    temp: int
    dt: float
    interval: int
    md_method: Literal['ase', 'openmm']


class USSettings(NamedTuple):
    """
    Settings struct for an Umbrella Sampling run.
    """
    temp: int           # temperature of sampling
    dt: float           # time difference
    interval: int       # ???
    n_windows: int      # ???
    init_ref: float     # where in the reaction profile to start sampling
    final_ref: float    # where in the reaction profile to end sampling
    ps: int             # ???


class MACEArchitectSettings(NamedTuple):
    """
    Settings for training a specific MACE architecture.
    """

    # model architecture params
    model_type: str                 # str, i.e. ScaleShiftMACE
    num_interactions: int           # number of message passing layers
    correlation: int                # body-order (n + 1), i.e. 3 = up to 4-body interactions
    max_L: int                      # tensor dimensions
    r_max: float                    # cutoff radius
    hidden_irreps: str = None       # also controls model size via controlling number of messages
    num_channels: int = None


class MACETrainSettings(NamedTuple):
    """
    Settings for a training run.
    """

    # training params
    energy_weight: float = 1.0
    forces_weight: float = 10.0
    lr: float = 0.001
    batch_size: int = 2
    max_num_epochs: int = 500
    E0s: str = None                        # reference isolated atom energies, str dict or 'average'

    # io params
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"

    # optional params
    swa: bool = False
    start_swa: int = None 
    ema: bool = True
    ema_decay: float = 0.99
    seed: int = 25
    restart_latest: bool = False
    scheduler_patience: int = None    # how many epochs to wait before LRScheduler reduces the learning rate
    patience: int = None              # early stopping patience, how many epochs without valid_loss improvement to stop training
    patience_warmup: int = None       # num epochs to wait before patience becomes active (helps with early volatile models)
    valid_batch_size: int = None      # validation batch size, should not affect convergence (except for slight noise), but can save on GPU MEM
    device: str = "cuda:0"             # resolved to 'cpu' at training time if unavailable
    enable_cueq: bool = False
    dtype: str = "float64"
    
    # multihead fine-tuning params
    multiheads_finetuning : bool = False      # multihead fine-tuning parameters (set as defaults, cf doc)
    heads : str = ""                          # when not same level of theory between replay and target OR different heads
    # when save level of theory
    num_samples_pt : int = 10000              # number of samples to use from the replay dataset
    #    filter_type_pt : str = "combinations" # filtering strategy for configurations
    subselect_pt : str = "random"             # method for subselection (random or fps)
    weight_pt : float = 1.0                   # weight for the pretraining head loss
    atomic_numbers : str = "[1,6,8]"          # which elements to keep in the replay dataset
    force_mh_ft_lr : bool = False             # lower learning rate + enables EMA during multiheads finetuning

    # lora params
    lora: bool = False
    lora_rank: int = 4
    lora_alpha: int = 1.0

    # logging params
    wandb_project: str = "default"

class BondDef(NamedTuple):
    """
    Definition of a specific bond between two atoms in a configuration.

    a1_id (int): id of the first atom which defines this bond 
    a2_id (int): id of the second atom which defined this bond
    bonded (bool): whether or not this bond is expected to exist (True) or not (False), 
                    as defined by the threshold
    threshold (float): the threshold within which the bond existing is defined.

    """
    a1_id: int
    a2_id: int
    bonded: bool
    threshold: float


class ConfigDef(NamedTuple):
    """
    Definition for a specific configuration based on a list of specific bond definitions.

    name (str): the name of the configuration (e.g. Reactant, Transition State, Product)
    bond_defs (list): a list of bond definitions which define this expected configuration.
    """
    name: str                   
    bond_defs: List[BondDef]


SINGLE_COLUMN_FIG_SIZE = (3.5, 2.5)
DOUBLE_COLUMN_FIG_SIZE = (7, 4)
DPI = 600
MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']


class PlotSettings(NamedTuple):
    """
    Standard Plot settings structure for matplotlib.
    """
    title: str = None
    x_label: str = None
    y_label: str = None
    use_log_y_scale: bool = False
    use_energy_log_scale: bool = False
    use_forces_log_scale: bool = False
    use_log_x_scale: bool = False
    xlim: tuple = None
    ylim: tuple = None
    integers_for_x = False
    integers_for_y = False
    energies_ylim: tuple = None
    forces_ylim: tuple = None
    highlight_region: tuple = None
    highlight_colour: str = 'green'
    line_width: float = 1.0
    grid_lines: bool = False
    horizontal_ref_line: float = None
    reverse_x: bool = False
    reverse_y: bool = False
    x_ticks: tuple = None # ()
    y_ticks: tuple = None
    x_num_dp: int = None
    y_num_dp: int = None

    # legend settings
    show_legend: bool = True
    legend_loc: str = "upper left"
    specific_legend_series: dict = None
    cust_legend_colours: list = None
    cust_legend_labels: list = None
    cust_legend_linestyles: list = None
    legend_fontsize: int = 4
    legend_ncols: int = 1

    # 
    colour_scheme: object = plt.cm.tab10
    colour_map_type: Literal['discrete', 'continuous'] = 'discrete'
    reference_indices: List[int] = []
    colour_dict: dict = None
    colour_bar_label: str = None
    colour_bar_range: tuple = None
    normalise_colour_bar: bool = False
    custom_markers: list = None
    custom_colour_list: list = None
    custom_linestyle_list: list = None
    custom_alpha_list: list = None
    custom_zorder_list: list = None
    cust_size_list: list = None
    cust_marker_linewidth: list = None
    point_size: float = 1.0

    num_base_models: int = None  # number of unique base models (for colours, if specified)
    num_base_model_comps: int = None  # number of comparisons per unique base model

    # overall figure sizes
    fig_size: tuple = SINGLE_COLUMN_FIG_SIZE
    dpi: int = DPI

    # for a bar plot
    bar_plot_mode: Literal['category', 'numerical'] = 'category'

