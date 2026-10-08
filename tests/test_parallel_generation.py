import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("generation", Path(__file__).parents[1] / "scripts/parallel_generate_videos.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class GenerationTests(unittest.TestCase):
    def test_unique_parallel_outputs_resume_and_changed_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, audio = root / "image.png", root / "audio.wav"
            image.touch()
            audio.touch()
            args = Namespace(output_root=root / "out", video_name="protected.mp4", config=root / "config.yaml", traceguard="traceguard", ffprobe="ffprobe")
            jobs = [("wam", "celeba", str(i), image, audio) for i in range(4)]
            names = []
            def subprocess_run(command, **kwargs):
                if command[0] == "traceguard":
                    output = Path(command[command.index("--output") + 1])
                    names.append(output.name)
                    output.write_bytes(b"fake video")
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=0, stdout="1.25")
            with patch.object(generation.subprocess, "run", side_effect=subprocess_run):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda job: generation.run_one(args, "3", job), jobs))
                self.assertEqual(results, ["PASS"] * 4)
                self.assertEqual(len(set(names)), 4)
                self.assertEqual(generation.run_one(args, "3", jobs[0]), "SKIP")
                new_audio = root / "new.wav"
                new_audio.touch()
                changed = (*jobs[0][:4], new_audio)
                self.assertEqual(generation.run_one(args, "3", changed), "PASS")

    def test_low_free_vram_keeps_jobs_pending_and_fails_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, audio, config = (root / name for name in ("image.png", "audio.wav", "config.yaml"))
            for path in (image, audio, config):
                path.touch()
            manifest = root / "celeba.jsonl"
            manifest.write_text(json.dumps(dict(sample_id="one", image=str(image), audio=str(audio))))
            args = Namespace(manifest=[manifest], output_root=root / "out", gpus="3", audio=None, config=config, video_name="protected.mp4", min_free_mib=16000)
            with patch.object(generation, "parse_args", return_value=args), patch.object(generation, "free_gpu_mib", return_value=1000), patch.object(generation, "run_one") as run:
                self.assertEqual(generation.main(), 1)
                run.assert_not_called()
