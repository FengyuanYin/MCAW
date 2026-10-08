import csv
import importlib.util
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("review", Path(__file__).parents[1] / "scripts/s19_review_server.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def test_concurrent_votes_and_suffix_video_range(self):
        with tempfile.TemporaryDirectory() as temp:
            review.REVIEW = Path(temp) / "review.csv"
            review.VIDEO_ROOT = Path(temp) / "videos"
            rows = [dict(domain="celeba", sample_id=f"sample-{i}", manual_ok="", reason="") for i in range(24)]
            with review.REVIEW.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            video = review.VIDEO_ROOT / "celeba/sample-0/clean.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"0123456789")
            server = ThreadingHTTPServer(("127.0.0.1", 0), review.Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                def vote(row):
                    row = dict(row, manual_ok="yes")
                    request = Request(base + "/api/review", data=json.dumps(row).encode(), headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=10) as response:
                        self.assertEqual(response.status, 200)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(vote, rows))
                self.assertEqual(sum(row["manual_ok"] == "yes" for row in review.read_csv(review.REVIEW)), 24)
                request = Request(base + "/video/celeba/sample-0", headers={"Range": "bytes=-3"})
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), b"789")
                    self.assertEqual(response.headers["Content-Range"], "bytes 7-9/10")
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
