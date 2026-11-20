import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from train_holds import ROOT, train


class TrainingTests(unittest.TestCase):
    def test_dataset_paths_and_best_weight_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "Final_Dataset"
            for split in ("train", "valid"):
                for kind in ("images", "labels"):
                    (dataset / split / kind).mkdir(parents=True)
            save_dir = root / "run"
            (save_dir / "weights").mkdir(parents=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"trained weights")
            observed = {}

            class Model:
                def __init__(self, weights):
                    observed["weights"] = weights
                    self.trainer = types.SimpleNamespace(save_dir=save_dir)

                def train(self, **kwargs):
                    observed.update(kwargs)
                    observed["data"] = yaml.safe_load(Path(kwargs["data"]).read_text())

            args = argparse.Namespace(data=ROOT / "data.yaml", dataset=dataset,
                                      weights="yolov8s.pt", epochs=50, imgsz=640,
                                      device="cpu", cpu_threads=None, output=root / "best_holds.pt")
            with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=Model)}):
                result = train(args)
            self.assertEqual(result.read_bytes(), b"trained weights")
            self.assertEqual(observed["data"]["path"], str(dataset.resolve()))
            self.assertEqual(observed["data"]["names"][0], "Crimp")
            self.assertEqual(observed["data"]["names"][1], "Jug")
            self.assertEqual(observed["data"]["names"][3], "Pocket")
            self.assertEqual(observed["epochs"], 50)

    def test_missing_dataset_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(data=ROOT / "data.yaml", dataset=Path(directory),
                                      weights="yolov8s.pt", epochs=50, imgsz=640,
                                      device="cpu", cpu_threads=None, output=Path(directory) / "best.pt")
            with self.assertRaisesRegex(FileNotFoundError, "train images"):
                train(args)

    def test_cpu_thread_setting_is_applied_after_trainer_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "valid"):
                for kind in ("images", "labels"):
                    (root / split / kind).mkdir(parents=True)
            (root / "weights").mkdir()
            (root / "weights" / "best.pt").write_bytes(b"weights")
            observed = []
            fake_torch = types.SimpleNamespace(set_num_threads=observed.append,
                                               get_num_threads=lambda: observed[-1])

            class Model:
                def __init__(self, _weights):
                    self.trainer = types.SimpleNamespace(
                        save_dir=root, device=types.SimpleNamespace(type="cpu")
                    )

                def add_callback(self, event, callback):
                    self.event, self.callback = event, callback

                def train(self, **kwargs):
                    self.callback(self.trainer)
                    observed.append(self.event)

            args = argparse.Namespace(data=ROOT / "data.yaml", dataset=root,
                                      weights="yolov8s.pt", epochs=50, imgsz=640,
                                      device="cpu", cpu_threads=20, output=root / "result.pt")
            modules = {"ultralytics": types.SimpleNamespace(YOLO=Model), "torch": fake_torch}
            with patch.dict(sys.modules, modules):
                train(args)
            self.assertEqual(observed, [20, "on_train_start"])


if __name__ == "__main__":
    unittest.main()
