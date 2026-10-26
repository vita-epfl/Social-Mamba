import copy
import logging
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from social_mamba.dataset import MultiPersonTrajPoseDataset
from social_mamba.evaluate import best_of_k_errors, evaluate_ade_fde, nba_reporting_scale
from social_mamba.model import create_model_UMamba
from social_mamba.utils.utils import load_config, save_config


ROOT = Path(__file__).resolve().parents[1]


class NBAReproducibilityTests(unittest.TestCase):
    def test_full_parameter_count_and_shared_blocks(self):
        config = yaml.safe_load((ROOT / "social_mamba/configs/UniHuMotion_NBA_full.yaml").read_text())
        model = create_model_UMamba(config, logging.getLogger("test"), "cpu")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1914030)
        for encoder, depth in ((model.local_former, 4), (model.global_former, 6)):
            self.assertEqual(len(encoder.layers), depth)
            self.assertTrue(all(block is encoder.layers[0] for block in encoder.layers))

    def test_independent_best_of_k_metrics(self):
        target = torch.zeros(1, 2, 2)
        prediction = torch.tensor([[[[0.0, 0.0], [2.0, 0.0]], [[4.0, 0.0], [0.0, 0.0]]]])
        ade, fde = best_of_k_errors(prediction, target)
        self.assertEqual(ade.item(), 1.0)
        self.assertEqual(fde.item(), 0.0)

    def test_vectorized_metrics_match_reference(self):
        generator = torch.Generator().manual_seed(0)
        prediction = torch.randn(7, 20, 20, 2, generator=generator)
        target = torch.randn(7, 20, 2, generator=generator)
        ade, fde = best_of_k_errors(prediction, target)
        for index in range(len(target)):
            errors = torch.stack([torch.norm(mode - target[index], dim=-1) for mode in prediction[index]])
            torch.testing.assert_close(ade[index], errors.mean(-1).min())
            torch.testing.assert_close(fde[index], errors[:, -1].min())

    def test_nba_full_and_legacy_units(self):
        self.assertEqual(nba_reporting_scale("NBA_full"), nba_reporting_scale("NBA"))
        self.assertAlmostEqual(nba_reporting_scale("NBA_full"), (1 / 0.3048) * (28 / 94))
        self.assertEqual(nba_reporting_scale("nba_score"), 1.0)
        self.assertEqual(nba_reporting_scale("nba_rebound"), 1.0)

    def test_full_rejects_eight_agent_cache(self):
        batch = (torch.zeros(1, 8, 30, 1, 4), torch.ones(1, 8, 30, 1), torch.zeros(1, 8).bool())
        config = {"DATA": {"train_datasets": ["NBA_full"]}, "TRAIN": {"output_track_size": 20}}
        with self.assertRaisesRegex(ValueError, "all 11 agents"):
            evaluate_ade_fde(None, [batch], 1, config, logging.getLogger("test"))

    def test_cache_ignores_non_pickle_files(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, SOCIAL_MAMBA_CACHE_DIR=temp):
            split = Path(temp) / "NBA_full/test"
            split.mkdir(parents=True)
            (split / "README.txt").write_text("not a cache")
            (split / "subdir").mkdir()
            with (split / "sample.pkl").open("wb") as stream:
                pickle.dump([(np.zeros((11, 30, 1, 4)), np.ones((11, 30, 1)))], stream)
            dataset = MultiPersonTrajPoseDataset("NBA_full", split="test")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset[0]), 1)

    def test_saved_config_includes_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            source = ROOT / "social_mamba/configs/UniHuMotion_NBA_full.yaml"
            output = tuple(str(Path(temp) / name) for name in ("logs", "checkpoint", "runs"))
            for directory in output:
                Path(directory).mkdir()
            with mock.patch("social_mamba.utils.utils.init_output_dirs", return_value=output):
                config = load_config(source)
            self.assertFalse((Path(output[1]) / "config.yaml").exists())
            config = copy.deepcopy(config)
            config["TRAIN"]["lr"] = 0.002
            config["MODEL"]["num_layers_local"] = 7
            save_config(config)
            saved = yaml.safe_load((Path(output[1]) / "config.yaml").read_text())
            self.assertEqual(saved["TRAIN"]["lr"], 0.002)
            self.assertEqual(saved["MODEL"]["num_layers_local"], 7)


if __name__ == "__main__":
    unittest.main()
