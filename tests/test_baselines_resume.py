import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from traceguard.baselines import _pgd_proxy, generate_baseline
from traceguard.engine.trainer import Trainer


class BaselineTests(unittest.TestCase):
    def test_proxy_moves_from_zero_gradient_reference_within_budget(self):
        image = torch.full((1, 3, 8, 8), 0.5)
        proxy = lambda clean, changed: SimpleNamespace(loss=-(changed-clean).square().mean())
        torch.manual_seed(7)
        first = _pgd_proxy(image, proxy, 0.05, steps=4)
        torch.manual_seed(7)
        second = _pgd_proxy(image, proxy, 0.05, steps=4)
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(float((first-image).square().mean()), 0.001)
        self.assertLessEqual(float((first-image).abs().max()), 0.050001)

    def test_joint_methods_require_and_dispatch_distinct_models(self):
        with self.assertRaises(ValueError):
            generate_baseline("naive_joint", None, None, None)
        naive, coupled = Mock(return_value="naive"), Mock(return_value="coupled")
        models = dict(naive_joint=naive, message_coupled=coupled)
        self.assertEqual(generate_baseline("naive_joint", None, None, None, joint_models=models), "naive")
        self.assertEqual(generate_baseline("message_coupled", None, None, None, joint_models=models), "coupled")
        naive.assert_called_once()
        coupled.assert_called_once()

    def test_resume_restores_optimizer_and_random_streams_on_cpu(self):
        trainer = Trainer.__new__(Trainer)
        trainer.protector = torch.nn.Linear(2, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.protector.parameters())
        trainer.scaler = Mock()
        trainer.protector(torch.ones(1, 2)).sum().backward()
        trainer.optimizer.step()
        random.seed(13)
        torch.manual_seed(13)
        state = dict(model=trainer.protector.state_dict(), optimizer=trainer.optimizer.state_dict(),
                     step=5, python_random=random.getstate(), torch_random=torch.get_rng_state())
        expected_python, expected_torch = random.random(), torch.rand(3)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            torch.save(state, path)
            with patch("torch.load", wraps=torch.load) as load:
                trainer.load_checkpoint(path)
                self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertEqual(trainer.step, 5)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertEqual(len(trainer.optimizer.state), 2)
