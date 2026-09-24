"""
IO file helpers.
"""
import os
import re
from contextlib import contextmanager
from typing import Callable, Dict, List, Literal, Union
import ase
from ase.io import read
from ase.io.extxyz import read_extxyz
import numpy as np
import glob
from react_ml_test_public.helpers.settings import (
    ANI_1XNR_MODEL_DIR,
    ANI_MODEL_LOADER,
    ANI_MODELS,
    GRAPH_PES_MODEL_LOADER,
    HARTREE_TO_EV,
    MACE_MODEL_LOADER,
    MACE_MODELS,
    resolve_model_dir,
)
from react_ml_test_public.helpers.logging import logger


def convert_npz_to_xyz():
    """
    ['R', 'E_true', 'E_predicted', 'F_true', 'F_predicted', 'Z', 'L', 'C', 'M', 'allow_pickle']
    (101, 190, 3)

    """

    data = np.load("../data/datasets/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca.npz")
    print(data.files)
    print(data['R'].shape)


def get_xyz_config_offsets(xyz_fpath: str, 
                           return_elements: bool = False) -> Union[list[int], tuple[list[int], list[list[str]]]]:
    """
    Get offsets of where all configurations begin and end in a .xyz file without explcitily loading Atoms objects through ase.
    This is useful for sampling individual configs from very large datasets.
    """

    offsets, elements = [], []
    with open(xyz_fpath, "r") as f:
        while True:

            # start of configuration
            start_offset = f.tell()
            first_line = f.readline()

            # EOF check
            if not first_line:
                break

            # number of atoms
            num_atoms = int(first_line.strip())
            _ = f.readline()  # Skip comment

            # skip atom lines
            config_elements = []
            for _ in range(num_atoms):
                atom_line = f.readline()
                config_elements.append(atom_line.split(' ')[0])

            end_offset = f.tell()  # End of configuration
            offsets.append((start_offset, end_offset))
            elements.append(config_elements)

    # add ability to load elements
    if return_elements:
        return offsets, elements
    else:
        return offsets


def save_dataset_batches(dataset_fp: str,
                         batch_size: int):
    """
    Save an input .xyz file into batches of batch_size, for seperate processing to save
    memory requirements of loading.
    """
    
    ds_name = get_name_from_path(dataset_fp)
    out_dir = '/'.join(dataset_fp.split('/')[0:-1])

    # get config offsets in file
    logger.info(f"Counting input configurations...")
    offsets = get_xyz_config_offsets(dataset_fp)
    n = len(offsets)
    logger.info(f"Total num configs in input dataset: {n}")

    # determine number of batches as given by batch size
    num_batches = int(n / batch_size)
    remainder = n % batch_size
    batch_lengths = [batch_size for _ in range(num_batches)]
    batch_lengths = batch_lengths + [remainder] if remainder != 0 else batch_lengths
    num_batches = len(batch_lengths)

    batched_offsets = [(offsets[sum(batch_lengths[:i])][0], 
                        offsets[sum(batch_lengths[:i+1]) - 1][1]) 
                        for i in range(num_batches)]

    with open(dataset_fp, "r") as f:

        for i, (start_offset, end_offset) in enumerate(batched_offsets):

            f.seek(start_offset)  # Jump to the configuration start
            frame_data = f.read(end_offset - start_offset)  # Read only this frame

            # write batch to file
            batch_fp = f"{out_dir}/{ds_name}_batch_{i}.xyz"
            with open(batch_fp, mode="w") as batch_file:
                batch_file.write(frame_data)
                logger.info(f"Saved dataset batch to to {batch_fp}...")

    logger.info(f"Successfully saved {ds_name} into {num_batches} batches...")


def flexible_ase_read(fpath: str,
                      cust_props_parser: Callable = None,
                      ) -> List[ase.Atoms]:
    """
    FLexible reader for ase .xyz or .extxyz files which handles different energy / force labels.
    """

    if cust_props_parser is not None:

        configs = read(fpath, ':')
        with open(fpath, 'r') as extxyz_file:
            custom_configs = read_extxyz(extxyz_file, index=slice(None), properties_parser=cust_props_parser)
            for config, custom_config in zip(configs, custom_configs):
                config.info['energy'] = custom_config.get_potential_energy()
    else:
        configs = read(fpath, ':')

    return configs


def get_xyz_file_len(fpath: str) -> int:
    """
    Counts the number of structures in an .xyz or .extxyz file
    """
    count = 0
    with open(fpath, 'r') as f:

        while True:
            num_atoms_line = f.readline()
            if not num_atoms_line:
                break  # EOF reached

            num_atoms = int(num_atoms_line.strip())  # Read number of atoms
            f.readline()  # Skip the comment line

            for _ in range(num_atoms):
                if not f.readline():
                    break  # Prevent reading past EOF

            count += 1

    return count


def orca_props_parser(prop_line: str) -> dict:
    match = re.search(r"E\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1)) * ase.units.Hartree}
    else:
        raise ValueError("No E <float> found in Props Line for orca xyz props parser...")


def qm_energy_label_props_parser(prop_line: str) -> dict:
    match = re.search(r"QM_energy\s*=\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1))}
    else:
        raise ValueError("No QM_energy = <float> found in Props Line for QM_energy label xyz props parser...")


def autode_props_parser(prop_line: str) -> dict:
    match = re.search(r"E\s*=\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1)) * ase.units.Hartree}
    else:
        raise ValueError("No E = <float> found in Props Line for autoDE xyz props parser...")


def hartree_props_parser(prop_line: str) -> dict:
    """
    Props parser for extxyz files labelled with ``REF_energy=<float>`` in Hartree, e.g. the
    reaction-profile files written with ``energy_units="Hartree"``. The energy is converted to eV
    (HARTREE_TO_EV) so it is on the same footing as the MLIP predictions.
    """
    match = re.search(r"REF_energy\s*=\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1)) * HARTREE_TO_EV}
    else:
        raise ValueError("No REF_energy = <float> found in Props Line for Hartree xyz props parser...")


def get_name_from_path(fpath: str):
    """
    Get the name of a file (i.e. dataset / model) from a path string.
    """
    fname = fpath.split('/')[-1]
    if '.xyz' in fname:
        return fname.split('.xyz')[0]
    elif '.model' in fname:
        return fname.split('.model')[0]
    elif '.info' in fname:
        return fname.split('.info')[0]
    elif '.csv' in fname:
        return fname.split('.csv')[0]
    elif '.npz' in fname:
        return fname.split('.npz')[0]
    elif '.out' in fname:
        return fname.split('.out')[0]
    else:
        raise ValueError('File extension not handled by path splitter.')


def get_dir_from_path(fpath: str):
    """
    Get the directory path of a file from a path string.
    """

    return fpath.rsplit('/', 1)[0]


def load_data_cache(cache_path: str):
    """
    Loads a .npz file, including pickled metadata.
    """

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        logger.info(f"Loaded npz data cache from {cache_path}...")
    else:
        logger.warning(f"NPZ cache file: {cache_path} does not exist, returning None!!!")
        return None
    return data


def save_npz_data_cache(npz_cache_path: str,
                        array_dict: Dict[str, np.ndarray],
                        meta: Dict = None):
    """
    Saves numpy arrays and pickled metadata as a .npz file.
    """

    # np.savez(npz_cache_path, **array_dict, meta=np.array(meta, dtype=object))
    npz_cache_dir = os.path.dirname(npz_cache_path)
    if npz_cache_dir:
        os.makedirs(npz_cache_dir, exist_ok=True)
    np.savez(npz_cache_path, **array_dict)
    logger.info(f"Saved npz data cache to {npz_cache_path}...")


def load_isolated_atom_energies(fpath: str):

    configs = flexible_ase_read(fpath)
    return {int(config.get_atomic_numbers()[0]): str(get_ase_atoms_energy(config)) for config in configs}


def get_ase_atoms_energy(config: ase.Atoms):
    """
    Load energy accounting for flexibly loaded xyz / extxyz files through ase.
    """ 
    try:
        energy = config.info['energy'] if config.info.get('energy') is not None else config.get_potential_energy()
        
    except RuntimeError:

        for energy_key in {'energy', 'QM_energy', 'REF_energy', 'E'}:
            energy = config.info.get(energy_key)
            if energy is not None:
                break
        if energy is None:
            raise RuntimeError("Energy label not found...")
        # config.info['energy'] = energy

    return energy


def get_ase_atoms_forces(config: ase.Atoms):
    """
    Load forces accounting for flexibly loaded xyz / extxyz files through ase.
    """
    try:
        forces = config.get_forces()
        # config.arrays['forces'] = forces
    except RuntimeError:

        for forces_key in {'forces', 'QM_forces', 'REF_forces'}:
            forces = config.arrays.get(forces_key)
            if forces is not None:
                break
        if forces is None:
            raise RuntimeError("Forces label not found...")
        # config.arrays['forces'] = forces

    return forces


def extract_num_atoms(text: str):
    patterns = [
        r"Number of atoms\s*[:=]\s*(\d+)",
        r"Total number of atoms\s*[:=]\s*(\d+)",
        r"n\s*atoms\s*[:=]\s*(\d+)",
        r"(\d+)\s+atoms",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Fallback: count atoms in the Cartesian coordinate table
    # This matches lines like:
    # C    0.000000   0.000000   0.000000
    coord_block = re.search(
        r"CARTESIAN COORDINATES.*?\n(?:[-=]+\n)?(.*?)(?:\n\s*\n|CARTESIAN COORDINATES|THE OPTIMIZATION|ORCA TERMINATED|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if coord_block:
        lines = coord_block.group(1).splitlines()
        atom_lines = [
            line for line in lines
            if re.match(r"^\s*[A-Z][a-z]?\s+[-+]?\d", line)
        ]
        if atom_lines:
            return len(atom_lines)

    return None


def check_orca_outputs(work_dir: str, input_name: str):
    succ, fail = [], []
    succ_num_atoms = {}
    fail_num_atoms = {}

    out_fpaths = glob.glob(os.path.join(work_dir, f"**/*{input_name}_*.out"), recursive=True)

    for out_fpath in out_fpaths:
        fname = get_name_from_path(out_fpath)
        run_id = int(re.search(r"(\d+)(?!.*\d)", fname).group()) + 1

        try:
            with open(out_fpath) as f:
                text = f.read()

            num_atoms = extract_num_atoms(text)

            if re.search(r"ORCA\s+TERMINATED\s+NORMALLY", text, re.IGNORECASE):
                succ.append(run_id)
                succ_num_atoms[run_id] = num_atoms
            else:
                fail.append(run_id)
                fail_num_atoms[run_id] = num_atoms

        except FileNotFoundError:
            print(f"{out_fpath} not found...")
            fail.append(run_id)
            fail_num_atoms[run_id] = None

    print("=== ORCA RUN STATS ===")
    print("Total Runs: ", len(out_fpaths))
    print("Success: ", len(succ))
    print("Success Jobs: ", sorted(succ))
    print("Success Num Atoms: ", {k: succ_num_atoms[k] for k in sorted(succ_num_atoms)})
    print("Fail: ", len(fail))
    print("Fail Jobs: ", sorted(fail))
    print("Fail Num Atoms: ", {k: fail_num_atoms[k] for k in sorted(fail_num_atoms)})
    print("=========================")


def format_orca_basis_block(basis_entries: List[tuple]) -> str:
    """
    Format the lines of an ORCA ``%basis`` block from ``(key, value)`` pairs.

    ORCA takes the basis set names in the block in quotation marks, e.g.
    ``AuxJ "aug-cc-pVTZ/JK"``, while the ``Decontract*`` switches are bare booleans:
    https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/basisset.html

    Parameters:
        basis_entries (list): (key, value) pairs, e.g. [("AuxJ", "aug-cc-pVTZ/JK")]

    Returns:
        (str): The block, e.g. ``%basis\\n  AuxJ "aug-cc-pVTZ/JK"\\nend\\n``
    """

    lines = ["%basis"]
    for key, value in basis_entries:
        lines.append(f"  {key} {_orca_block_value(value)}")
    lines.append("end")

    return "\n".join(lines) + "\n"


def _orca_block_value(value) -> str:
    """Render one ORCA block value: a bare boolean, else the value in quotation marks."""

    if isinstance(value, bool):
        return "true" if value else "false"

    # a boolean written as a string (e.g. from a YAML "false") is also left unquoted
    text = str(value)
    if text.lower() in ("true", "false"):
        return text.lower()

    return f'"{text}"'


def insert_orca_input_block(input_fpath: str, block: str) -> None:
    """
    Insert a block into a generated ORCA input file, after the leading ``!`` line.

    autode's ORCA wrapper (``autode.wrappers.ORCA.ORCA.generate_input_for``) writes the
    ``!`` simple-input keywords followed by a fixed set of blocks, with no hook for extra
    block input. A block autode cannot express - e.g. the orbital / auxiliary basis slots
    of ``%basis``, which have no equivalent in mlptrain's ``orca_keywords`` - is therefore
    inserted into the file once autode has generated it.

    Parameters:
        input_fpath (str): Path to the ORCA .inp file to modify
        block (str): The block to insert, e.g. from ``format_orca_basis_block``
    """

    with open(input_fpath, 'r') as f:
        lines = f.readlines()

    # the ! line is written first, so the block goes directly after it (ORCA reads block
    # input in any order, so inserting at the top of the file is fine either way)
    insert_at = 1 if lines and lines[0].lstrip().startswith('!') else 0
    lines.insert(insert_at, block if block.endswith('\n') else block + '\n')

    with open(input_fpath, 'w') as f:
        f.writelines(lines)

    logger.info(f"Inserted block into {input_fpath}")


def resolve_cuda_device(cuda_device: str = "cuda:0") -> str:
    """Resolve a requested torch device onto one that actually exists on this machine.

    ``cuda_device`` is the device requested by a script default or an input YAML
    (e.g. 'cuda:0' (the default), 'cuda:1', 'cuda' or 'cpu'). The returned device is
    always usable:

      * an explicit 'cpu' request is kept as-is,
      * a CUDA request falls back to 'cpu' when CUDA is unavailable, when the requested
        GPU index is out of range (e.g. 'cuda:1' on a single-GPU machine), when the
        device string cannot be understood, or when the GPU cannot be initialised.

    When a CUDA device resolves it is also set as the current torch CUDA device, so
    implicit 'cuda' usage agrees with it.

    ``torch`` is imported lazily so that importing this module does not pull in the
    PyTorch stack.
    """
    import torch

    # a missing / blank / 'None' request means the default GPU
    requested = "cuda:0" if cuda_device is None else str(cuda_device).strip()
    if requested == "" or requested.lower() == "none":
        requested = "cuda:0"

    # CPU explicitly requested
    if requested.lower() == "cpu":
        logger.info("Using CPU (cuda_device: 'cpu')...")
        return "cpu"

    if not requested.lower().startswith("cuda"):
        logger.warning(f"Unrecognised cuda_device: {requested!r}, using CPU instead...")
        return "cpu"

    # 'cuda' with no index means the default device, 'cuda:N' means device N
    index = 0
    if requested.lower() != "cuda":
        try:
            index = int(requested.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            logger.warning(f"Unrecognised cuda_device: {requested!r}, using CPU instead...")
            return "cpu"

    if not torch.cuda.is_available():
        logger.warning(f"Requested device {requested!r} but CUDA is not available, using CPU...")
        return "cpu"

    num_devices = torch.cuda.device_count()
    if not 0 <= index < num_devices:
        logger.warning(f"Requested device {requested!r} but only {num_devices} CUDA "
                       f"device(s) found, using CPU...")
        return "cpu"

    resolved_device = f"cuda:{index}"
    try:
        torch.cuda.set_device(index)
        device_name = torch.cuda.get_device_name(index)
    except Exception as e:  # e.g. driver / device initialisation failures
        logger.warning(f"Could not use {resolved_device}: {e} \n... Using CPU instead...")
        return "cpu"

    logger.info(f"Using device {resolved_device} ({device_name})...")
    return resolved_device


def check_cuda_setup(cuda_device: str = "cuda:0") -> str:
    """Log the device that will be used and return it.

    ``cuda_device`` is resolved first (see ``resolve_cuda_device``), so a requested GPU
    that is not present on this machine falls back to 'cpu' rather than failing later
    when a calculator is built. A small test tensor is then moved onto the resolved
    device and its location logged.

    Returns the resolved device string ('cuda:N' or 'cpu'), which callers should pass
    on to their calculators.

    ``torch`` is imported lazily so that importing this module does not pull in the
    PyTorch stack.
    """
    import torch

    resolved_device = resolve_cuda_device(cuda_device)
    logger.info(f"CUDA is available: {torch.cuda.is_available()}")

    # Create a tensor and move it onto the resolved device
    tensor = torch.tensor([1.0, 2.0, 3.0]).to(torch.device(resolved_device))

    # Check where the tensor ended up
    logger.info(f"Tensor device: {tensor.device}")
    return resolved_device


@contextmanager
def _jit_load_on_device(device: str):
    """Force ``torch.jit.load`` onto ``device`` for the duration of the block.

    MACE models embed compiled e3nn ``CodeGenMixin`` submodules, and e3nn restores them
    with a bare ``torch.jit.load(buffer)`` - no ``map_location`` (e3nn
    ``util/codegen/_mixin.py``, still the case in e3nn 0.4.4 and upstream). Without a
    ``map_location`` that call reproduces the device the model was saved on, which is
    normally a GPU, so loading such a model on a machine where that GPU is not visible
    raises ``RuntimeError: No CUDA GPUs are available`` - even when the calculator was
    asked to run on the CPU, and no outer ``map_location`` (passed to ``torch.load``, or
    MACE's own ``extract_load``) can reach the nested call.

    Injecting ``map_location`` here is the same handling MACE uses when it loads
    TorchScript itself (``mace.tools.scripts_utils.load_from_json``), and it also keeps
    the submodules on the requested device when a non-default GPU index is selected.

    ``torch`` is imported lazily so that importing this module does not pull in the
    PyTorch stack.
    """
    import torch

    original_jit_load = torch.jit.load

    def jit_load(f, map_location=None, *args, **kwargs):
        return original_jit_load(f, map_location=map_location or device, *args, **kwargs)

    torch.jit.load = jit_load
    try:
        yield
    finally:
        torch.jit.load = original_jit_load


def load_model_calculator(model_name: str,
                          device='cuda:0',
                          model_loader: Literal['MACE', 'ani', 'graph-pes'] = None,
                          enable_cueq: bool = False,
                          dtype: Literal["", "float32", "float64"] = "",
                          head: str = "Default",
                          model_dir: str = None):
    """
    Flexible model loader for different types of model based on model_name (i.e. ACE models/ MACE models / ANI models / transformer models etc.)

    ``device`` is resolved first (see ``resolve_cuda_device``), so the default 'cuda:0' -
    and any 'cuda:N' that is not present on this machine - falls back to 'cpu'.

    ``model_name`` is the model name alone: its '*.model' file is looked for in
    ``model_dir`` first and then anywhere below it, so a model kept in a subdirectory
    (e.g. ``models/fine_tuned/``) loads without the subdirectory being given in the
    input yaml. The first name match wins.

    ``model_dir`` defaults to ``MODEL_DIR`` (see ``resolve_model_dir``), which a YAML
    input can override with its ``model_dir`` field to load models from another
    repository's ``models/`` directory.
    """

    # resolve the requested device before building any calculator
    device = resolve_cuda_device(device)

    # a YAML input that sets model_dir loads its models from there instead of MODEL_DIR
    model_dir = resolve_model_dir(model_dir)

    # resolve the model fpath, searching model_dir recursively so a model kept in a
    # subdirectory (e.g. models/fine_tuned/) can be named by its name alone -
    # the first match wins, with a model in model_dir itself taking precedence
    model_fpath = f"{model_dir}/{model_name}.model"
    if not os.path.exists(model_fpath):
        model_fpaths = sorted(glob.glob(f"{model_dir}/**/{model_name}.model", recursive=True))
        if model_fpaths:
            model_fpath = model_fpaths[0]
            logger.info(f"Found {model_name} at {model_fpath}...")

    try:
        with _jit_load_on_device(device):

            # attempt to load model_loader from model name
            if model_loader is None:

                if model_name in MACE_MODELS:
                    model_loader = MACE_MODEL_LOADER
                elif model_name in ANI_MODELS:
                    model_loader = ANI_MODEL_LOADER
                elif any(sub_model_name in model_name for sub_model_name in MACE_MODELS + ['MACE', 'EGRET']):
                    model_loader = MACE_MODEL_LOADER
                elif any(sub_model_name in model_name for sub_model_name in ANI_MODELS):
                    model_loader = ANI_MODELS
                else:
                    model_loader = MACE_MODEL_LOADER  # default to MACE model loader!

            if model_loader == MACE_MODEL_LOADER:

                # mace model loader (imported lazily to keep io.py importable without MACE)
                from mace.calculators import MACECalculator

                return MACECalculator(model_paths=model_fpath,
                                      device=device,
                                      enable_cueq=enable_cueq,
                                      default_dtype=dtype,
                                      head=head
                                    )

            elif model_loader == ANI_MODEL_LOADER:

                # ani model loader (torchani imported lazily)
                import torchani

                # read constants from constant file and construct AEV computer.
                const_fpath = f'{ANI_1XNR_MODEL_DIR}/{model_name}/rHCNO-5.2R_32-3.5A_a8-4.params'  # noqa: E501
                consts = torchani.neurochem.Constants(const_fpath)
                aev_computer = torchani.AEVComputer(**consts)

                # read self energies and construct energy shifter.
                sae_fpath = f'{ANI_1XNR_MODEL_DIR}/{model_name}/sae_linfit.dat'  # noqa: E501
                energy_shifter = torchani.neurochem.load_sae(sae_fpath)

                # single model
                # NB: named distinctly from the model_dir argument above - this is the
                # torchani networks directory, not the *.model search root
                ani_networks_dir = f'{ANI_1XNR_MODEL_DIR}/{model_name}/train0/networks'  # noqa: E501
                model = torchani.neurochem.load_model(consts.species, ani_networks_dir)

                # create pipeline for computing energies
                nnp = torchani.nn.Sequential(aev_computer, model, energy_shifter)

                # get torchani ASE calculator (calculates in Hartrees)
                return torchani.ase.Calculator(consts.species, nnp)

            elif model_loader == GRAPH_PES_MODEL_LOADER:

                from graph_pes.models import load_model
                import torch

                model = load_model(model_fpath)
                model.to(dtype=torch.float64)
                return model.ase_calculator()
            else:
                raise NameError(f"Model loader: {model_loader} is not valid.")

    except Exception as e:
        logger.warning(f"{model_name} was unable to be loaded because of error: \n{e} \n... Returning None for calc instead...")
        return None


def substitute_yaml_variables(value: dict, variables: dict):
    """
    Little recursive function to fill in YAML variables based on its own defined variable names.

    USAGE: yaml_config = substitute_yaml_variables(yaml_config, yaml_config)
    """
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda match: str(variables.get(match.group(1), match.group(0))),
            value,
        )
    elif isinstance(value, list):
        return [substitute_yaml_variables(item, variables) for item in value]
    elif isinstance(value, dict):
        return {
            key: substitute_yaml_variables(item, variables)
            for key, item in value.items()
        }
    return value