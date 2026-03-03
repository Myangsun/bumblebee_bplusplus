#!/usr/bin/env python3
"""
run.py — Thin orchestrator for the bumblebee pipeline.

Dispatches to pipeline modules so each step is individually importable
and directly runnable without going through this file.

Usage
-----
# Data pipeline
python run.py collect
python run.py analyze
python run.py analyze --split-dir GBIF_MA_BUMBLEBEES/prepared_split
python run.py prepare
python run.py split

# Augmentation
python run.py augment --method copy_paste --targets Bombus_sandersoni
python run.py augment --method synthetic --species Bombus_ashtoni Bombus_sandersoni --count 450

# Training
python run.py train --type simple --data-dir GBIF_MA_BUMBLEBEES/prepared_split
python run.py train --type hierarchical --dataset raw

# Evaluation
python run.py evaluate --type metrics
python run.py evaluate --type bioclip

# Full pipeline
python run.py all

Individual scripts can also be run directly:
    python pipeline/train/simple.py --data-dir ... --output-dir ...
    python pipeline/augment/synthetic.py --species Bombus_ashtoni --count 30
"""

import argparse
import sys


def _cmd_collect(args):
    from pipeline.collect import main as _main
    sys.argv = ["pipeline/collect.py"]
    _main()


def _cmd_analyze(args):
    from pipeline.analyze import run, run_split_analysis
    save_plot = not getattr(args, "no_plot", False)
    output_dir = getattr(args, "output_dir", None) or "RESULTS"
    reference_dir = getattr(args, "reference_dir", None)
    if args.split_dir:
        run_split_analysis(split_dir=args.split_dir, output_dir=output_dir,
                           save_plot=save_plot, reference_dir=reference_dir)
    else:
        data_dir = getattr(args, "data_dir", None) or "GBIF_MA_BUMBLEBEES"
        run(data_dir=data_dir, output_dir=output_dir, save_plot=save_plot)


def _cmd_prepare(args):
    from pipeline.prepare import main as _main
    sys.argv = ["pipeline/prepare.py"]
    _main()


def _cmd_split(args):
    from pipeline.split import main as _main
    sys.argv = ["pipeline/split.py"]
    _main()


def _cmd_augment(args):
    if args.method == "copy_paste":
        from pipeline.augment.copy_paste import main as _main
        argv = ["pipeline/augment/copy_paste.py"]
        if args.targets:
            argv += ["--targets"] + args.targets
        if args.dataset_root:
            argv += ["--dataset-root", args.dataset_root]
        if args.sam_checkpoint:
            argv += ["--sam-checkpoint", args.sam_checkpoint]
        if args.count:
            argv += ["--per-class-count", str(args.count)]
        sys.argv = argv
        _main()
    elif args.method == "synthetic":
        from pathlib import Path
        from pipeline.augment.synthetic import run as _run
        _run(
            species=args.species or None,
            count=args.count or 450,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
    else:
        print(f"Unknown augmentation method: {args.method}")
        print("Available: copy_paste, synthetic")
        sys.exit(1)


def _cmd_train(args):
    if args.type == "simple":
        from pipeline.train.simple import main as _main
        argv = ["pipeline/train/simple.py"]
        if args.data_dir:
            argv += ["--data-dir", args.data_dir]
        if args.output_dir:
            argv += ["--output-dir", args.output_dir]
        if args.backbone:
            argv += ["--backbone", args.backbone]
        if args.epochs:
            argv += ["--epochs", str(args.epochs)]
        sys.argv = argv
        _main()
    elif args.type == "hierarchical":
        from pipeline.train.hierarchical import main as _main
        argv = ["pipeline/train/hierarchical.py"]
        if args.dataset:
            argv += ["--dataset", args.dataset]
        if args.train_only:
            argv += ["--train-only"]
        if args.test_only:
            argv += ["--test-only"]
        sys.argv = argv
        _main()
    elif args.type == "hierarchical_focus":
        from pipeline.train.hierarchical_focus import main as _main
        argv = ["pipeline/train/hierarchical_focus.py"]
        if args.dataset:
            argv += ["--dataset", args.dataset]
        if args.focus_species:
            argv += ["--focus-species"] + args.focus_species
        if args.train_only:
            argv += ["--train-only"]
        if args.test_only:
            argv += ["--test-only"]
        sys.argv = argv
        _main()
    else:
        print(f"Unknown training type: {args.type}")
        print("Available: simple, hierarchical, hierarchical_focus")
        sys.exit(1)


def _cmd_evaluate(args):
    if args.type == "metrics":
        from pipeline.evaluate.metrics import main as _main
        argv = ["pipeline/evaluate/metrics.py"]
        if args.models:
            argv += ["--models"] + args.models
        if args.test_dir:
            argv += ["--test-dir", args.test_dir]
        if args.suffix:
            argv += ["--suffix", args.suffix]
        sys.argv = argv
        _main()
    elif args.type == "bioclip":
        from pipeline.evaluate.bioclip import main as _main
        argv = ["pipeline/evaluate/bioclip.py"]
        if args.data_root:
            argv += ["--data-root", args.data_root]
        if args.split:
            argv += ["--split", args.split]
        sys.argv = argv
        _main()
    else:
        print(f"Unknown evaluation type: {args.type}")
        print("Available: metrics, bioclip")
        sys.exit(1)


def _cmd_all(args):
    print("Running full pipeline: collect → analyze → prepare → split → train → evaluate")
    _cmd_collect(args)

    class _AnalyzeArgs:
        split_dir = None
        output_dir = None
        no_plot = False
        data_dir = None

    _cmd_analyze(_AnalyzeArgs())
    _cmd_prepare(args)
    _cmd_split(args)

    # Default: simple training on prepared_split
    class _TrainArgs:
        type = "simple"
        data_dir = "GBIF_MA_BUMBLEBEES/prepared_split"
        output_dir = "RESULTS/simple_model"
        backbone = None
        epochs = None
        dataset = None
        train_only = False
        test_only = False

    _cmd_train(_TrainArgs())

    class _EvalArgs:
        type = "metrics"
        models = None
        test_dir = None
        suffix = "gbif"
        data_root = None
        split = "train"

    _cmd_evaluate(_EvalArgs())


def main():
    parser = argparse.ArgumentParser(
        description="Bumblebee pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── collect ──────────────────────────────────────────────────────────────
    sub.add_parser("collect", help="Download GBIF bumblebee images")

    # ── analyze ──────────────────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="Analyze dataset distribution")
    p_analyze.add_argument("--data-dir", help="Raw dataset directory")
    p_analyze.add_argument("--split-dir", help="Analyze a train/valid/test split directory")
    p_analyze.add_argument("--output-dir", help="Output directory for reports and plots")
    p_analyze.add_argument("--no-plot", action="store_true", help="Skip distribution plot")
    p_analyze.add_argument("--reference-dir", help="Use species order from this baseline split dir")

    # ── prepare ──────────────────────────────────────────────────────────────
    sub.add_parser("prepare", help="YOLO-crop images and create 80/20 train/valid split")

    # ── split ────────────────────────────────────────────────────────────────
    sub.add_parser("split", help="Reorganize into 70/15/15 train/valid/test split")

    # ── augment ──────────────────────────────────────────────────────────────
    p_aug = sub.add_parser("augment", help="Augment dataset (copy_paste or synthetic)")
    p_aug.add_argument("--method", required=True, choices=["copy_paste", "synthetic"],
                       help="Augmentation method")
    p_aug.add_argument("--targets", nargs="+", help="Species targets (copy_paste)")
    p_aug.add_argument("--species", nargs="+", help="Species to generate (synthetic)")
    p_aug.add_argument("--count", type=int, help="Images per species")
    p_aug.add_argument("--dataset-root", help="Dataset root path (copy_paste)")
    p_aug.add_argument("--sam-checkpoint", help="SAM checkpoint path (copy_paste)")
    p_aug.add_argument("--output-dir", help="Output directory (synthetic)")

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a classification model")
    p_train.add_argument("--type", required=True, choices=["simple", "hierarchical", "hierarchical_focus"],
                         help="Training type")
    p_train.add_argument("--data-dir", help="Data directory (simple)")
    p_train.add_argument("--output-dir", help="Output directory")
    p_train.add_argument("--backbone", choices=["resnet18", "resnet50", "resnet101"],
                         help="Backbone (simple)")
    p_train.add_argument("--epochs", type=int, help="Number of epochs")
    p_train.add_argument("--dataset", help="Dataset type (hierarchical): raw, cnp, synthetic, ...")
    p_train.add_argument("--train-only", action="store_true", help="Skip test step (hierarchical)")
    p_train.add_argument("--test-only", action="store_true", help="Skip train step (hierarchical)")
    p_train.add_argument("--focus-species", nargs="+",
                         help="Species for C1b checkpoint (hierarchical_focus only)")

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate trained models")
    p_eval.add_argument("--type", required=True, choices=["metrics", "bioclip"],
                        help="Evaluation type")
    p_eval.add_argument("--models", nargs="+", help="Model keys to test (metrics)")
    p_eval.add_argument("--test-dir", help="Override test directory (metrics)")
    p_eval.add_argument("--suffix", default="gbif", help="Output file suffix (metrics)")
    p_eval.add_argument("--data-root", help="Dataset root (bioclip)")
    p_eval.add_argument("--split", default="train", help="Dataset split (bioclip)")

    # ── all ──────────────────────────────────────────────────────────────────
    sub.add_parser("all", help="Run the full pipeline end to end")

    args = parser.parse_args()

    dispatch = {
        "collect": _cmd_collect,
        "analyze": _cmd_analyze,
        "prepare": _cmd_prepare,
        "split": _cmd_split,
        "augment": _cmd_augment,
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "all": _cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
