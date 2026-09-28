"""Local-only video review UI for S19; writes votes to clean_review.csv."""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data/formal_eval/clean_review.csv"
ORDER = ROOT / "data/formal_eval/clean_review_order.csv"
VIDEO_ROOT = ROOT / "outputs/formal_eval/clean"
VOTE_LOCK = threading.Lock()

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>S19 干净视频人工复核</title>
<style>
body{font:16px system-ui,sans-serif;max-width:950px;margin:24px auto;padding:0 16px;background:#f5f6f8;color:#222}
header,section{background:white;padding:16px 20px;margin:12px 0;border-radius:10px}
video{width:100%;max-height:570px;background:#111}button{padding:10px 16px;margin:6px;cursor:pointer}
input{padding:9px;width:54%}.ok{background:#d7f5df}.bad{background:#ffe0df}
#status{font-weight:600}small{color:#555}
</style>
<header><h2>S19 干净视频人工复核</h2><div id="summary"></div></header>
<section><div id="status"></div><small id="details"></small><p><video id="video" controls preload="metadata"></video></p>
<button class="ok" id="yes">通过 Y</button><button class="bad" id="no">不通过 N</button>
<input id="reason" placeholder="不通过原因（必填）"><p><button id="previous">上一条 ←</button>
<button id="next">下一条 →</button><button id="unreviewed">下一条未复核</button></p>
<small>标准：视频可播放、单人脸、脸部稳定、口型与固定音频相符。请听声音并完整观看；不要依据保护方法效果筛选。</small>
</section>
<script>
let items=[], index=0;
const el=id=>document.getElementById(id);
function draw(){let x=items[index];if(!x)return;el('status').textContent=`${index+1}/${items.length} · ${x.domain} · ${x.sample_id} · ${x.manual_ok||'未复核'}`;
el('details').textContent=`时长 ${x.duration_seconds}s · ${x.width}×${x.height}`;
el('video').src=`/video/${encodeURIComponent(x.domain)}/${encodeURIComponent(x.sample_id)}`;
el('video').load();el('reason').value=x.reason||'';
let yes=items.filter(z=>z.manual_ok==='yes'),no=items.filter(z=>z.manual_ok==='no');
el('summary').textContent=`CelebA 通过 ${yes.filter(z=>z.domain==='celeba').length}/40；TH1KH 通过 ${yes.filter(z=>z.domain==='th1kh').length}/40；不通过 ${no.length}；未复核 ${items.length-yes.length-no.length}`;}
async function refresh(){items=await (await fetch('/api/items')).json();let first=items.findIndex(x=>!x.manual_ok);index=first>=0?first:0;draw()}
async function vote(value){let x=items[index],reason=el('reason').value.trim();if(value==='no'&&!reason){alert('请填写不通过原因');return}
let r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:x.domain,sample_id:x.sample_id,manual_ok:value,reason})});
if(!r.ok){alert(await r.text());return}x.manual_ok=value;x.reason=reason;let next=items.findIndex((z,i)=>i>index&&!z.manual_ok);if(next>=0)index=next;else index=Math.min(index+1,items.length-1);draw()}
el('yes').onclick=()=>vote('yes');el('no').onclick=()=>vote('no');
el('previous').onclick=()=>{index=Math.max(0,index-1);draw()};el('next').onclick=()=>{index=Math.min(items.length-1,index+1);draw()};
el('unreviewed').onclick=()=>{let j=items.findIndex(x=>!x.manual_ok);if(j>=0){index=j;draw()}};
document.onkeydown=e=>{if(e.target.tagName==='INPUT')return;if(e.key==='y')vote('yes');if(e.key==='n')vote('no');if(e.key==='ArrowLeft')el('previous').click();if(e.key==='ArrowRight')el('next').click()};
refresh();
</script></html>"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Handler(BaseHTTPRequestHandler):
    def send_data(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self.send_data(PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/items":
            votes = {(row["domain"], row["sample_id"]): row for row in read_csv(REVIEW)}
            order = read_csv(ORDER)
            for row in order:
                vote = votes[(row["domain"], row["sample_id"])]
                row["manual_ok"] = vote["manual_ok"]
                row["reason"] = vote["reason"]
            return self.send_data(json.dumps(order, ensure_ascii=False).encode(), "application/json")
        match = re.fullmatch(r"/video/(celeba|th1kh)/([A-Za-z0-9_.-]+)", unquote(path))
        if not match:
            return self.send_data(b"Not found", "text/plain", 404)
        domain, sample_id = match.groups()
        if (domain, sample_id) not in {(row["domain"], row["sample_id"]) for row in read_csv(REVIEW)}:
            return self.send_data(b"Not found", "text/plain", 404)
        video = VIDEO_ROOT / domain / sample_id / "clean.mp4"
        if not video.is_file():
            return self.send_data(b"Video missing", "text/plain", 404)
        size = video.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match_range = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match_range:
                return self.send_data(b"Invalid range", "text/plain", 416)
            if not match_range[1] and not match_range[2]:
                return self.send_data(b"Invalid range", "text/plain", 416)
            if match_range[1]:
                start = int(match_range[1])
                if match_range[2]:
                    end = min(int(match_range[2]), size - 1)
            else:
                suffix = int(match_range[2])
                if suffix <= 0:
                    return self.send_data(b"Invalid range", "text/plain", 416)
                start = max(0, size - suffix)
            if start > end or start >= size:
                return self.send_data(b"Range not satisfiable", "text/plain", 416)
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with video.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def do_POST(self) -> None:
        if self.path != "/api/review":
            return self.send_data(b"Not found", "text/plain", 404)
        try:
            if int(self.headers.get("Content-Length", "0")) > 4096:
                raise ValueError("Request too large")
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            domain, sample_id = data["domain"], data["sample_id"]
            choice, reason = data["manual_ok"], str(data.get("reason", "")).strip()
            if choice not in {"yes", "no"} or (choice == "no" and not reason):
                raise ValueError("Use yes/no and provide a reason for no")
            with VOTE_LOCK:
                rows = read_csv(REVIEW)
                matches = [row for row in rows if row["domain"] == domain and row["sample_id"] == sample_id]
                if len(matches) != 1:
                    raise ValueError("Unknown sample")
                matches[0]["manual_ok"] = choice
                matches[0]["reason"] = reason
                temp = REVIEW.with_suffix(".csv.tmp")
                with temp.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temp.replace(REVIEW)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.send_data(str(exc).encode(), "text/plain", 400)
        self.send_data(b'{"saved": true}', "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not REVIEW.is_file() or not ORDER.is_file():
        raise FileNotFoundError("Run s19_prepare_review.py first")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"S19 review UI on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
