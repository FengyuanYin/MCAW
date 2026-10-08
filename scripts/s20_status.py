"""Print concise status for the current S20 remote pilot and image batch."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def marker(name: str) -> str:
    path = Path("outputs") / f"{name}.exit"
    return path.read_text().strip() if path.is_file() else "RUNNING"


def main() -> None:
    root = Path("outputs/s20_video_pilot2k")
    statuses = [json.loads(path.read_text()) for path in root.rglob("status.json")]
    counts = Counter("PASS" if row.get("generated") else "FAIL" for row in statuses)
    methods = Counter(row["method"] for row in statuses if row.get("generated"))
    full_root = Path("outputs/s20_video_40x2_proxy")
    full_statuses = [json.loads(path.read_text()) for path in full_root.rglob("status.json")]
    full_counts = Counter("PASS" if row.get("generated") else "FAIL" for row in full_statuses)
    full_methods = Counter(row["method"] for row in full_statuses if row.get("generated"))
    resume_log = Path("outputs/s20_video_40x2_proxy_resume.log")
    resume_counts = Counter()
    if resume_log.is_file():
        for line in resume_log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"\b(PASS|FAIL|SKIP) \{", line)
            if match:
                resume_counts[match.group(1)] += 1
    print(json.dumps({
        "images_40x2_exit": marker("s20_images_40x2_proxy"),
        "video_pilot_initial_exit": marker("s20_video_pilot2k"),
        "video_pilot_retry_exit": marker("s20_video_retry2k"),
        "video_score_exit": marker("s20_video_score2k"),
        "video_jobs_expected": 28,
        "video_jobs_completed": len(statuses),
        "video_jobs_passed": counts["PASS"],
        "video_jobs_failed": counts["FAIL"],
        "passed_by_method": dict(sorted(methods.items())),
        "full_proxy_initial_exit": marker("s20_video_40x2_proxy"),
        "full_proxy_resume_exit": marker("s20_video_40x2_proxy_resume"),
        "full_proxy_score_exit": marker("s20_video_score_40x2_proxy"),
        "full_proxy_jobs_expected": 560,
        "full_proxy_jobs_with_status": len(full_statuses),
        "full_proxy_jobs_passed": full_counts["PASS"],
        "full_proxy_jobs_failed": full_counts["FAIL"],
        "full_proxy_passed_by_method": dict(sorted(full_methods.items())),
        "resume_attempt_counts": dict(sorted(resume_counts.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
