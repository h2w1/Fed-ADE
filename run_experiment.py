#!/usr/bin/env python3
"""Command-line runner for Fed-ADE experiments.

This runner replaces the old sim_ADE_*.ipynb files. It injects runtime
configuration into the converted ADE.py scripts and executes one experiment.
"""
from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONFIGS = {
    ("label", "cifar10"): {
        "script": ROOT / "experiments" / "label_shift" / "CIFAR" / "ADE.py",
        "workdir": ROOT / "experiments" / "label_shift" / "CIFAR",
        "lr_min": 1e-5,
        "lr_max": 1e-3,
        "dist_key": "dist_type",
        "default_dist": "uniform",
    },
    ("label", "tinyimagenet"): {
        "script": ROOT / "experiments" / "label_shift" / "TinyImageNet" / "ADE.py",
        "workdir": ROOT / "experiments" / "label_shift" / "TinyImageNet",
        "lr_min": 5e-6,
        "lr_max": 1e-4,
        "dist_key": "dist",
        "default_dist": "uniform",
    },
    ("covariate", "cifar10c"): {
        "script": ROOT / "experiments" / "covariate_shift" / "CIFAR" / "ADE.py",
        "workdir": ROOT / "experiments" / "covariate_shift" / "CIFAR",
        "lr_min": 1e-6,
        "lr_max": 1e-4,
        "dist_key": "dist_type",
        "default_dist": "uniform",
    },
    ("covariate", "cifar100c"): {
        "script": ROOT / "experiments" / "covariate_shift" / "CIFAR100" / "ADE.py",
        "workdir": ROOT / "experiments" / "covariate_shift" / "CIFAR100",
        "lr_min": 1e-5,
        "lr_max": 1e-4,
        "dist_key": "dist_type",
        "default_dist": "uniform",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Fed-ADE post-adaptation experiment.")
    parser.add_argument("--shift-type", choices=["label", "covariate"], required=True)
    parser.add_argument("--dataset", choices=["cifar10", "tinyimagenet", "cifar10c", "cifar100c"], required=True)
    parser.add_argument("--shift", choices=["lin", "sin", "squ", "ber"], default="lin", help="Temporal shift schedule.")
    parser.add_argument("--seed-index", type=int, default=0, help="Index into the seed list used by the original experiments.")
    parser.add_argument("--dist", default="uniform", help="Pretraining distribution tag. Release pack includes only 'uniform'.")
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-max", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="Override communication rounds per timestep.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override number of online timesteps.")
    parser.add_argument("--num-client", type=int, default=None, help="Override number of federated clients.")
    parser.add_argument("--sample-per-step", type=int, default=None)
    parser.add_argument("--lambda", dest="lamda", type=float, default=300)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value, e.g., 0. Omit to use the current environment.")
    parser.add_argument("--cifar-c-dir", default=None, help="Path to CIFAR-10-C or CIFAR-100-C directory for covariate-shift experiments.")
    parser.add_argument("--tiny-imagenet-dir", default=None, help="Path to tiny-imagenet-200 root directory.")
    parser.add_argument("--pretrained-path", default=None, help="Override pretrained model checkpoint path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    key = (args.shift_type, args.dataset)
    if key not in CONFIGS:
        raise SystemExit(f"Unsupported combination: shift_type={args.shift_type}, dataset={args.dataset}")

    cfg = CONFIGS[key]
    if args.dist != "uniform":
        raise SystemExit("This release pack includes only uniform pretrained checkpoints. Use --dist uniform.")

    globals_to_inject = {
        "now_running": args.seed_index,
        "shift": args.shift,
        "LR_MIN": args.lr_min if args.lr_min is not None else cfg["lr_min"],
        "LR_MAX": args.lr_max if args.lr_max is not None else cfg["lr_max"],
        "lamda": args.lamda,
        "cuda_visible_devices": args.gpu,
    }
    globals_to_inject[cfg["dist_key"]] = args.dist
    globals_to_inject["dist_type"] = args.dist
    globals_to_inject["dist"] = args.dist

    if args.rounds is not None:
        globals_to_inject["ROUNDS"] = args.rounds
    if args.timesteps is not None:
        globals_to_inject["T"] = args.timesteps
    if args.num_client is not None:
        globals_to_inject["NUM_CLIENT"] = args.num_client
    if args.sample_per_step is not None:
        globals_to_inject["sample_per_step"] = args.sample_per_step
    if args.cifar_c_dir is not None:
        globals_to_inject["cifar_c_dir"] = str(Path(args.cifar_c_dir).expanduser().resolve())
    if args.tiny_imagenet_dir is not None:
        tiny_root = Path(args.tiny_imagenet_dir).expanduser().resolve()
        globals_to_inject["tiny_imagenet_train_dir"] = str(tiny_root / "train")
        globals_to_inject["tiny_imagenet_val_dir"] = str(tiny_root / "val")
    if args.pretrained_path is not None:
        globals_to_inject["pretrained_path"] = str(Path(args.pretrained_path).expanduser().resolve())

    old_cwd = Path.cwd()
    os.chdir(cfg["workdir"])
    try:
        result = runpy.run_path(str(cfg["script"]), init_globals=globals_to_inject)
    finally:
        os.chdir(old_cwd)

    if "final_acc" in result:
        print(f"final_acc: {result['final_acc']:.4f}")
    if "elapsed_time" in result:
        print(f"elapsed_time_sec: {result['elapsed_time']:.2f}")


if __name__ == "__main__":
    main()
