"""Train a hold detector using an extracted Kaggle Final_Dataset directory."""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def train(args: argparse.Namespace) -> Path:
    config_path = args.data.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    dataset = args.dataset.resolve() if args.dataset else (
        config_path.parent / config["path"]
    ).resolve()
    for split in ("train", "val"):
        images = dataset / config[split]
        if not images.is_dir():
            raise FileNotFoundError(f"{split} images directory not found: {images}")
        labels = images.parent / "labels"
        if not labels.is_dir():
            raise FileNotFoundError(f"{split} labels directory not found: {labels}")
    config["path"] = str(dataset)

    from ultralytics import YOLO

    # Ultralytics expects a YAML path, and the temporary config carries an
    # absolute dataset path independent of the current working directory.
    with tempfile.TemporaryDirectory(prefix="climbing-data-") as temp_dir:
        resolved_config = Path(temp_dir) / "data.yaml"
        resolved_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        model = YOLO(args.weights)
        if args.cpu_threads is not None:
            import torch

            def set_cpu_threads(trainer):
                if trainer.device.type == "cpu":
                    torch.set_num_threads(args.cpu_threads)
                    print(f"CPU training threads: {torch.get_num_threads()}", flush=True)

            model.add_callback("on_train_start", set_cpu_threads)
        model.train(
            data=str(resolved_config),
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            project=str(ROOT / "artifacts" / "runs" / "detect"),
        )
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Training did not produce best weights: {best}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.output)
    return args.output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--dataset", type=Path, help="Path to extracted Final_Dataset")
    parser.add_argument("--weights", default=str(ROOT / "models" / "yolov8s.pt"),
                        help="Initial detection weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="PyTorch CPU training threads (default: Ultralytics setting)")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "best_holds.pt")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 1:
        parser.error("--epochs and --imgsz must be positive")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    print(f"Best weights saved to {train(args)}")


if __name__ == "__main__":
    main()
