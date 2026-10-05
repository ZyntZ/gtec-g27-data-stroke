"""Command-line entry point."""
import argparse



def main(argv=None):
    parser = argparse.ArgumentParser(description="Stroke Rehab: reproducible EEG baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download, verify and extract organizer data")
    download.add_argument("--archive", default="data/stroke-rehab.rar")
    download.add_argument("--data-dir", default="data/stroke-rehab")
    inventory = sub.add_parser("inventory", help="Validate recordings and write provenance")
    inventory.add_argument("--data-dir", default="data/stroke-rehab")
    inventory.add_argument("--output", default="results/data_inventory.csv")
    run = sub.add_parser("run", help="Fit training runs; score held-out test runs")
    run.add_argument("--data-dir", default="data/stroke-rehab")
    run.add_argument("--output-dir", default="results")
    run.add_argument("--seed", type=int, default=27)
    run.add_argument("--folds", type=int, default=5)
    audit = sub.add_parser("audit", help="Predeclared early/late CSP sensitivity check")
    audit.add_argument("--data-dir", default="data/stroke-rehab")
    audit.add_argument("--output", default="results/timing_audit.csv")
    diagnostics = sub.add_parser("diagnose", help="Training-only pre-cue and temporal CV controls")
    diagnostics.add_argument("--config", default="configs/diagnostics.json")
    diagnostics.add_argument("--data-dir", default=None)
    diagnostics.add_argument("--output", default=None)
    diagnostics.add_argument("--seed", type=int, default=None)
    diagnostics.add_argument("--folds", type=int, default=None)
    diagnostics.add_argument("--purge", type=int, default=None)
    causal = sub.add_parser("causal-train", help="Replay and score the fixed causal training-only decoder")
    causal.add_argument("--config", default="configs/causal_training.json")
    causal.add_argument("--data-dir", default=None)
    causal.add_argument("--output", default=None)
    nested = sub.add_parser("nested-train", help="Nested blocked CV for fixed causal spatial candidates")
    nested.add_argument("--config", default="configs/spatial_nested.json")
    nested.add_argument("--data-dir", default=None)
    nested.add_argument("--output-dir", default=None)
    calibrate = sub.add_parser("calibrate-spatial", help="Fit trusted, local models on training runs only")
    calibrate.add_argument("--config", default="configs/spatial_nested.json")
    calibrate.add_argument("--data-dir", default=None)
    calibrate.add_argument("--output-dir", default="data/models")
    latency = sub.add_parser("latency-audit", help="Training-only, fixed-CSP feedback-timing sensitivity")
    latency.add_argument("--config", default="configs/spatial_latency.json")
    latency.add_argument("--data-dir", default=None)
    latency.add_argument("--output", default=None)
    control = sub.add_parser("controls", help="Training-only pre-cue and label-permutation checks")
    control.add_argument("--config", default="configs/negative_controls.json")
    control.add_argument("--data-dir", default=None)
    control.add_argument("--output", default=None)
    for name, text, output in (
            ("montage-check", "Test which channel layout each recording follows",
             "results/montage_check.csv"),
            ("compare", "Filter-bank CSP versus Riemannian decoder, same trials and windows",
             "results/model_comparison.csv"),
            ("decision-time", "Causal accuracy against decision time, both models",
             "results/decision_time.csv"),
            ("lateralize", "Descriptive C3/C4 beta change with per-file channel layout",
             "results/lateralization.csv"),
            ("transfer", "Decode each session with a model from the patient's other session",
             "results/transfer.csv")):
        extra = sub.add_parser(name, help=text)
        extra.add_argument("--data-dir", default="data/stroke-rehab")
        extra.add_argument("--output", default=output)
    plot_time = sub.add_parser("plot-decision-time", help="Visualize saved decision-time scores")
    plot_time.add_argument("--csv", default="results/decision_time.csv")
    plot_time.add_argument("--output", default="results/decision_time.png")
    plot_time.add_argument("--theme", choices=("light", "dark"), default="light")
    forward = sub.add_parser("forward-calibration", help="Training-only causal forward validation at fixed calibration budgets")
    forward.add_argument("--config", default="configs/forward_calibration.json")
    forward.add_argument("--data-dir", default="data/stroke-rehab")
    forward.add_argument("--output-dir", default="results/forward_calibration")
    forward_plot = sub.add_parser("plot-forward-calibration", help="Plot the training-only calibration audit")
    forward_plot.add_argument("--csv", default="results/forward_calibration/forward_calibration_summary.csv")
    forward_plot.add_argument("--output", default="results/forward_calibration/forward_calibration.png")
    plot = sub.add_parser("plot", help="Visualize saved session scores")
    plot.add_argument("--csv", default="results/session_results.csv")
    plot.add_argument("--output", default="results/session_accuracy.png")
    args = parser.parse_args(argv)
    if args.command == "download":
        from .download import download_and_extract
        download_and_extract(args.archive, args.data_dir)
    elif args.command == "inventory":
        from .inventory import build_inventory
        print(f"Validated {len(build_inventory(args.data_dir, args.output))} MAT files")
    elif args.command == "run":
        from .experiment import evaluate_all
        evaluate_all(args.data_dir, args.output_dir, seed=args.seed,
                     n_splits=args.folds)
    elif args.command == "audit":
        from .audit import run_audit
        print(run_audit(args.data_dir, args.output))
    elif args.command == "diagnose":
        from .diagnostics import run_diagnostics
        import json
        from pathlib import Path
        config = json.loads(Path(args.config).read_text())
        rows = run_diagnostics(
            args.data_dir if args.data_dir is not None else config["data_dir"],
            args.output if args.output is not None else config["output"],
            seed=args.seed if args.seed is not None else config["seed"],
            n_splits=args.folds if args.folds is not None else config["folds"],
            purge=args.purge if args.purge is not None else config["purge_trials"],
        )
        print(f"Wrote {len(rows)} training-only diagnostic scores")
    elif args.command == "causal-train":
        import json
        from pathlib import Path
        from .causal_experiment import run_causal_training
        config = json.loads(Path(args.config).read_text())
        run_causal_training(
            args.data_dir if args.data_dir is not None else config["data_dir"],
            args.output if args.output is not None else config["output"],
            chunk_samples=config["chunk_samples"],
            window=tuple(config["window_seconds"]),
            n_splits=config["folds"], purge=config["purge_trials"])
    elif args.command in ("nested-train", "calibrate-spatial"):
        import json
        from pathlib import Path
        from .nested import export_calibrations, run_nested_training
        config = json.loads(Path(args.config).read_text())
        data_dir = args.data_dir if args.data_dir is not None else config["data_dir"]
        if args.command == "nested-train":
            run_nested_training(data_dir,
                args.output_dir if args.output_dir is not None else config["output_dir"],
                chunk_samples=config["chunk_samples"],
                window=tuple(config["window_seconds"]),
                outer_folds=config["outer_folds"], inner_folds=config["inner_folds"],
                purge=config["purge_trials"])
        else:
            export_calibrations(data_dir, args.output_dir,
                chunk_samples=config["chunk_samples"],
                window=tuple(config["window_seconds"]),
                n_splits=config["inner_folds"], purge=config["purge_trials"])
    elif args.command == "latency-audit":
        import json
        from pathlib import Path
        from .nested import run_latency_audit
        config = json.loads(Path(args.config).read_text())
        rows = run_latency_audit(
            args.data_dir if args.data_dir is not None else config["data_dir"],
            args.output if args.output is not None else config["output"],
            window_starts=config["window_start_s"],
            stops=tuple(config["window_stops_s"]),
            chunk_samples=config["chunk_samples"],
            folds=config["folds"], purge=config["purge_trials"])
        print(f"Saved {len(rows)} exploratory training-only latency checks")
    elif args.command == "controls":
        import json
        from pathlib import Path
        from .controls import run_controls
        config = json.loads(Path(args.config).read_text())
        run_controls(
            args.data_dir if args.data_dir is not None else config["data_dir"],
            args.output if args.output is not None else config["output"],
            permutations=config["permutations"], seed=config["seed"],
            folds=config["folds"], purge=config["purge_trials"],
            chunk_samples=config["chunk_samples"])
    elif args.command == "montage-check":
        from .montage import run_montage_check
        run_montage_check(args.data_dir, args.output)
    elif args.command == "compare":
        from .comparison import run_comparison
        run_comparison(args.data_dir, args.output)
    elif args.command == "decision-time":
        from .comparison import run_decision_time
        run_decision_time(args.data_dir, args.output)
    elif args.command == "lateralize":
        from .lateralization import run_lateralization
        print(f"Wrote {len(run_lateralization(args.data_dir, args.output))} rows")
    elif args.command == "transfer":
        from .transfer import run_transfer
        run_transfer(args.data_dir, args.output)
    elif args.command == "forward-calibration":
        from .forward_calibration import run_forward_calibration
        import json
        from pathlib import Path
        design = json.loads(Path(args.config).read_text(encoding="utf-8"))
        predictions, summary = run_forward_calibration(
            args.data_dir, args.output_dir,
            budgets=design["budgets"], stops=design["decision_times_s"],
            validation_start=design["validation_start_0based"],
            validation_stop=design["validation_stop_0based"],
            window_s=design["window_s"])
        print(f"Training-only: {len(predictions)} decisions, {len(summary)} session/model/time/budget rows")
    elif args.command == "plot-forward-calibration":
        from .forward_plot import plot_forward_calibration
        print(plot_forward_calibration(args.csv, args.output))
    elif args.command == "plot-decision-time":
        from .plotting import plot_decision_time
        print(plot_decision_time(args.csv, args.output, theme=args.theme))
    else:
        from .plotting import plot_results
        print(plot_results(args.csv, args.output))


if __name__ == "__main__":
    main()
