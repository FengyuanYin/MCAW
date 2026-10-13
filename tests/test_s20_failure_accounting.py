import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("score", Path(__file__).parents[1] / "scripts/s20_score_videos.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class FailureAccountingTests(unittest.TestCase):
    def test_strict_mode_and_failure_aware_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(method="wam", domain="th1kh", sample_id=str(i), image=f"image{i}", audio="audio", message="01") for i in range(2)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
            for i, row in enumerate(rows):
                folder = root / "videos/wam/th1kh" / str(i)
                folder.mkdir(parents=True)
                (folder / "status.json").write_text(json.dumps(dict(image=row["image"], audio="audio", generated=i == 0, exit_code=0 if i == 0 else 2)))
                if i == 0:
                    (folder / "protected.mp4").write_bytes(b"video")
                else:
                    (folder / "run.log").write_text("#face is invalid: 0")
            argv = ["score", "--manifest", str(manifest), "--video-root", str(root / "videos"), "--output-root", str(root / "scores")]
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                score.main()
            config = Mock()
            config.get.side_effect = lambda key, default: default
            decoded = SimpleNamespace(message="01", frames=[dict(detection_score=0.8)], detection_score=0.8, valid_frame_ratio=1.0)
            decoder = Mock()
            decoder.decode_video.return_value = decoded
            with patch("sys.argv", argv+["--allow-generation-failures"]), patch.object(score, "load_config", return_value=config), patch.object(score, "resolve_device", return_value="cpu"), patch.object(score, "build_wam"), patch.object(score, "TemporalDecoder", return_value=decoder):
                self.assertEqual(score.main(), 0)
            summary = json.loads((root / "scores/summary.json").read_text())
            self.assertEqual(summary["expected_jobs"], 2)
            self.assertEqual(summary["total_videos"], 1)
            self.assertEqual(summary["generation_failures"], 1)
            self.assertEqual(summary["generation_success_rate"], 0.5)
            self.assertEqual(summary["by_method_domain"][0]["mean_bit_accuracy"], 1.0)
            failure = json.loads((root / "scores/generation_failures.jsonl").read_text())
            self.assertEqual(failure["reason"], "face_preprocessing_failed")
            decoder.decode_video.assert_called_once()
