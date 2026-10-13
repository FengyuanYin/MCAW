import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "hallo_entry", Path(__file__).parents[1] / "src/traceguard/adapters/hallo_inference.py"
)
hallo_entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hallo_entry)


class HalloEntryTests(unittest.TestCase):
    def test_verified_entry_preserves_absolute_paths_and_project_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            hallo = root / "third_party/hallo"
            config = Mock(root=root)
            config.resolve.return_value = hallo
            output = Path("outputs/relative-hallo/video.mp4")
            expected = output.resolve()
            # Replace mkdir to keep this mocked launch free of output artifacts.
            with patch.object(hallo_entry, "_runtime_config", return_value=expected.parent / "hallo_runtime.yaml") as runtime, patch.object(hallo_entry.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, patch.object(Path, "mkdir"):
                hallo_entry.generate_hallo_video(config, Path("image.png"), Path("audio.wav"), output)
            runtime.assert_called_once_with(config, expected.parent / "hallo_runtime.yaml")
            command = run.call_args.args[0]
            self.assertEqual(command[1:3], ["-m", "traceguard.adapters.hallo_verified"])
            self.assertEqual(command[command.index("--hallo-root") + 1], str(hallo))
            self.assertEqual(command[command.index("--config") + 1], str(expected.parent / "hallo_runtime.yaml"))
            self.assertEqual(command[command.index("--output") + 1], str(expected))
            self.assertEqual(run.call_args.kwargs["cwd"], root)
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], str(hallo))


if __name__ == "__main__":
    unittest.main()
