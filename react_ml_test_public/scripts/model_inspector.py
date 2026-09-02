"""Script for inspecting model architecture using visualisation tools:"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: UP035

import torch
from torch import nn

from react_ml_test_public.helpers.settings import MODEL_DIR, OUT_DIR

DEVICE = torch.device("cuda:0")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, but GPU loading was requested.")

def load_model_file(path: Path) -> Any:
    return torch.load(path, map_location=DEVICE, weights_only=False)

# def load_model_file(path: Path) -> Any:
#     def _map_location(storage, loc):
#         return storage.cuda(0)

#     return torch.load(path, map_location=_map_location, weights_only=False)

def extract_model(obj: Any) -> Optional[nn.Module]:
    if isinstance(obj, nn.Module):
        return obj

    if isinstance(obj, dict):
        for key in ("model", "net", "module"):
            if isinstance(obj.get(key), nn.Module):
                return obj[key]

    return None


def extract_state_dict(obj: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(obj, dict):

        for key in ("state_dict", "model_state_dict", "net_state_dict"):
            val = obj.get(key)

            if (
                isinstance(val, dict)
                and val
                and all(torch.is_tensor(v) for v in val.values())
            ):
                return val

        # Raw state_dict
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj

    return None


def count_module_stats(model: nn.Module) -> Dict[str, Any]:
    params = list(model.parameters())
    buffers = list(model.buffers())

    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    frozen_params = total_params - trainable_params

    layer_type_counts: Dict[str, int] = {}

    all_modules = list(model.modules())[1:]
    num_layers = len(all_modules)

    for m in all_modules:
        cls = m.__class__.__name__
        layer_type_counts[cls] = layer_type_counts.get(cls, 0) + 1

    return {
        "kind": "nn.Module",
        "num_layers": num_layers,
        "num_buffers": len(buffers),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "layer_type_counts": "; ".join(
            f"{k}:{v}"
            for k, v in sorted(
                layer_type_counts.items(),
                key=lambda x: (-x[1], x[0]),
            )
        ),
    }


def count_state_dict_stats(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Any]:

    total_params = sum(t.numel() for t in state_dict.values())

    return {
        "kind": "state_dict",
        "num_layers": "",
        "num_buffers": "",
        "total_params": total_params,
        "trainable_params": "",
        "frozen_params": "",
        "layer_type_counts": "",
        "num_tensors": len(state_dict),
    }


def summarize_file(path: str) -> Dict[str, Any]:
    
    obj = load_model_file(path)

    model = extract_model(obj)

    if model is not None:
        stats = count_module_stats(model)

    else:
        state_dict = extract_state_dict(obj)

        if state_dict is not None:
            stats = count_state_dict_stats(state_dict)

        else:
            stats = {
                "kind": type(obj).__name__,
                "num_layers": "",
                "num_buffers": "",
                "total_params": "",
                "trainable_params": "",
                "frozen_params": "",
                "layer_type_counts": "",
            }

    stats["file"] = str(path)
    stats["name"] = path.stem

    return stats


def print_table(rows: List[Dict[str, Any]], cols: List[str]) -> None:
    widths = {c: len(c) for c in cols}
    str_rows = []

    for row in rows:
        srow = {}
        for c in cols:
            s = str(row.get(c, ""))
            srow[c] = s
            widths[c] = max(widths[c], len(s))
        str_rows.append(srow)

    sep = " | "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    line = "-+-".join("-" * widths[c] for c in cols)

    print(header)
    print(line)
    for row in str_rows:
        print(sep.join(row[c].ljust(widths[c]) for c in cols))


if __name__ == "__main__":

    # ============================================================
    # INPUTS
    # ============================================================

    # pretrained models
    model_names = [
        "GO-MACE-23",
        "MACE-OFF23_medium", 
        "MACE-MP-0b3_medium",
        'MACE-omol-0-extra-large-1024',
    ]
    out_name = "pretrained_models"

    # ============================================================
    # PROCESS
    # ============================================================

    model_fpaths = [f"{MODEL_DIR}/{model_name}.model" for model_name in model_names]
    summary_csv = f"{OUT_DIR}/model_info/{out_name}_summary.csv"
    layer_counts_csv = f"{OUT_DIR}/model_info/{out_name}_layer_counts.csv"
    rows = [summarize_file(Path(p)) for p in model_fpaths]

    # Summary table without layer counts
    summary_cols = [
        "name",
        "kind",
        "num_layers",
        "total_params",
        "trainable_params",
        "frozen_params",
        "num_buffers",
        "file",
    ]

    print("\nMODEL SUMMARY\n")
    print_table(rows, summary_cols)

    # Build wide layer-type table
    # for row in rows:
    #     print(row.get("layer_type_counts", {}))
    #     print([s.split(':') for s in row.get("layer_type_counts", {}).split(';')])

    all_layer_type_counts = [
                [(layer_name.strip(), int(count.strip()))
                for layer_name, count in [s.split(':') for s in row.get("layer_type_counts", {}).split(';')]]
                for row in rows
            ]

    print("All layer type counts: \n", all_layer_type_counts)
    layer_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        out_row = {"name": row["name"]}
        for layer_name, count in all_layer_type_counts[i]:
            out_row[layer_name] = count
        layer_rows.append(out_row)

    # print([x[0] for x in all_layer_type_counts[0]])
    layer_cols = ["name"] + list(set([x[0] for layer_type_counts in all_layer_type_counts for x in layer_type_counts ]))

    print("\nLAYER TYPE COUNTS\n")
    if all_layer_type_counts is not None:
        print_table(layer_rows, layer_cols)
    else:
        print("No layer types found.")



    # ============================================================
    # SAVE CSVs
    # ============================================================

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in summary_cols})

    with open(layer_counts_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=layer_cols)
        writer.writeheader()
        writer.writerows(layer_rows)

    print(f"\nSaved summary CSV to: {Path(summary_csv).resolve()}")
    print(f"Saved layer-count CSV to: {Path(layer_counts_csv).resolve()}")
