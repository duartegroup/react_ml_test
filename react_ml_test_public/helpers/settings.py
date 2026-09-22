"""
Config file containing file paths for data and outputs, and other general settings.
"""
import os
import platform
from typing import List, Literal, NamedTuple

import matplotlib.pyplot as plt

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

MACE_MODEL_LOADER = 'MACE'
ANI_MODEL_LOADER = 'ani'
GRAPH_PES_MODEL_LOADER = 'graph-pes'
MODEL_LOADERS = [MACE_MODEL_LOADER, ANI_MODEL_LOADER, GRAPH_PES_MODEL_LOADER]

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
NPZ_DIR = f"{OUT_DIR}/npz"
SAMPLES_OUT_DIR = f"{OUT_DIR}/samples"
DESCRIPTORS_DIR = f"{OUT_DIR}/descriptors"
SIMILARITIES_DIR = f"{OUT_DIR}/similarities"
CHEMISCOPE_DIR = f"{OUT_DIR}/chemiscope"


def resolve_out_dirs(out_dir: str = None) -> tuple[str, str, str]:
    """
    Resolve an output root directory into the subdirectories used by the workflow scripts.

    A YAML input can set ``out_dir`` to write its outputs somewhere other than the default
    ``OUT_DIR`` (e.g. when running from a directory other than the repository root); passing
    None falls back to ``OUT_DIR``. The subdirectory layout matches the one defined above,
    i.e. plots under ``<out_dir>/plt`` and CSVs under ``<out_dir>/csv``.

    :param out_dir: output root directory, or None to use OUT_DIR.
    :return: (out_dir, plot_dir, csv_dir)
    """
    out_dir = OUT_DIR if out_dir is None else str(out_dir)
    return out_dir, f"{out_dir}/plt", f"{out_dir}/csv"


def resolve_data_path(fpath: str, data_dir: str = None) -> str:
    """
    Resolve a YAML input's data path against a data root directory.

    Absolute paths, and paths that already resolve from the current working directory (so
    existing inputs that spell out e.g. ``data/reaction_data/...`` keep working), are returned
    unchanged; anything else is taken to be relative to ``data_dir`` (default ``DATA_DIR``).
    A path that resolves under neither is returned as written, so a typo is reported against the
    path the input actually spells out.

    :param fpath: data file path from the input YAML.
    :param data_dir: data root directory, or None to use DATA_DIR.
    :return: the resolved path.
    """
    if fpath is None or os.path.isabs(fpath) or os.path.exists(fpath):
        return fpath
    data_fpath = os.path.join(DATA_DIR if data_dir is None else str(data_dir), fpath)
    return data_fpath if os.path.exists(data_fpath) else fpath

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

