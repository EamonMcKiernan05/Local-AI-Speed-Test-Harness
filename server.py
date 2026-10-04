#!/usr/bin/env python3
"""model-bench web server, stdlib only. Serves the UI and runs one battery at a time.

    python3 server.py --port 8090 --bind 0.0.0.0
"""
import argparse
import json
import threading
import time
import traceback
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import bench

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
UI = HERE / "ui.html"
LOCK = threading.Lock()
JOB = {"cur": None}
DOWNLOAD_OK = {"meta.json", "results.jsonl", "summary.csv", "summary.md"}
CTYPE = {"meta.json": "application/json", "results.jsonl": "application/x-ndjson",
         "summary.csv": "text/csv", "summary.md": "text/markdown; charset=utf-8"}


class Job:
    def __init__(self, cfg):
        self.cfg = cfg
        self.status = "running"
        self.rows = []
        self.error = None
        self.run_dir = None
        self.started = time.time()
        self.finished = None
        self.stop_flag = False
        self.total = len(cfg["depths"]) * cfg["reps"] + 1  # +1 = warm-up
        self.done = 0
        self.current = "warm-up"

    def pub(self):
        return {"id": Path(self.run_dir).name if self.run_dir else None,
                "status": self.status,
                "cfg": {k: self.cfg.get(k) for k in ("url", "label", "backend", "gen", "reps", "depths")},
                "rows": self.rows[-500:],
                "error": self.error, "run_dir": self.run_dir,
                "started": self.started, "finished": self.finished,
                "done": self.done, "total": self.total, "current": self.current}


def worker(job):
    def on_row(row):
        with LOCK:
            job.rows.append(row)
            if row.get("kind") in ("arm", "skipped", "warmup"):
                job.done += 1
            if row.get("kind") == "warmup":
                job.current = "warm-up"
            elif row.get("kind") == "arm":
                job.current = "%s · rep %s" % (bench.fmt_depth(row["depth_k"]), row.get("rep"))
            if row.get("kind") == "skipped":
                job.current = "skip %s" % bench.fmt_depth(row["depth_k"])
    try:
        rd = bench.run_job(job.cfg, on_row=on_row, should_stop=lambda: job.stop_flag)
        with LOCK:
            job.run_dir = str(rd)
            job.status = "aborted" if job.stop_flag else "done"
    except Exception:
        with LOCK:
            job.status = "error"
            job.error = traceback.format_exc()[-900:]
    finally:
        with LOCK:
            job.finished = time.time()


def list_runs(limit=40):
    out = []
    if not RUNS.exists():
        return out
    dirs = sorted((d for d in RUNS.iterdir() if d.is_dir()), reverse=True)[:limit]
    for d in dirs:
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            continue
        out.append({"id": d.name, "label": meta.get("label", d.name),
                    "endpoint": meta.get("endpoint", ""), "backend": meta.get("backend", ""),
                    "model": (meta.get("detected") or {}).get("model", ""),
                    "started": (meta.get("started_utc") or "")[:16],
                    "status": "aborted" if meta.get("aborted") else ("done" if meta.get("finished_utc") else "incomplete"),
                    "files": [f for f in ("summary.md", "summary.csv", "results.jsonl", "meta.json")
                              if (d / f).exists()]})
    return out


def parse_start(data):
    url = bench.normalize_base(data.get("url") or "")
    if not url.startswith("http"):
        raise ValueError("endpoint must start with http:// or https://")
    battery = data.get("battery") or "standard"
    if battery == "standard":
        depths = bench.DEPTHS_STANDARD
    elif battery == "quick":
        depths = bench.DEPTHS_QUICK
    else:
        depths = bench.parse_depths(data.get("custom") or "")
        if not depths:
            raise ValueError("custom battery needs depths, e.g. 10k, 100k")
    try:
        gen = max(16, min(8192, int(data.get("gen") or bench.GEN_DEFAULT)))
        reps = max(1, min(10, int(data.get("reps") or 1)))
    except (TypeError, ValueError):
        raise ValueError("gen/reps must be numbers")
    return {"url": url, "label": (data.get("label") or "").strip()[:80],
            "model": (data.get("model") or "").strip()[:120],
            "backend": data.get("backend") or "", "depths": depths, "gen": gen, "reps": reps,
            "timeout": 2400, "out_root": str(RUNS)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except Exception:
            return {}

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            if not UI.exists():
                self._send(500, "ui.html missing", "text/plain")
                return
            self._send(200, UI.read_text(), "text/html; charset=utf-8")
            return
        if u.path == "/api/state":
            with LOCK:
                cur = JOB["cur"].pub() if JOB["cur"] else None
            self._json({"job": cur, "runs": list_runs()})
            return
        if u.path == "/api/detect":
            qs = urllib.parse.parse_qs(u.query)
            url = bench.normalize_base((qs.get("url") or [""])[0])
            if not url:
                self._json({"ok": False, "error": "no url"})
                return
            try:
                self._json({"ok": True, **bench.detect(url)})
            except Exception as e:
                self._json({"ok": False, "error": str(e)[:300]})
            return
        if u.path.startswith("/runs/"):
            parts = u.path.split("/")
            if len(parts) == 4 and parts[2] and parts[2] == Path(parts[2]).name and parts[3] in DOWNLOAD_OK:
                p = RUNS / parts[2] / parts[3]
                if p.is_file():
                    ctype = CTYPE.get(parts[3], "application/octet-stream")
                    extra = {}
                    if parts[3] != "summary.md":
                        extra["Content-Disposition"] = 'attachment; filename="%s-%s"' % (parts[2], parts[3])
                    self._send(200, p.read_bytes(), ctype, extra)
                    return
            self._send(404, "not found", "text/plain")
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/start":
            with LOCK:
                running = JOB["cur"] is not None and JOB["cur"].status == "running"
            if running:
                self._json({"error": "a battery is already running"}, 409)
                return
            try:
                cfg = parse_start(self._read_body())
            except ValueError as e:
                self._json({"error": str(e)}, 400)
                return
            job = Job(cfg)
            with LOCK:
                JOB["cur"] = job
            threading.Thread(target=worker, args=(job,), daemon=True).start()
            self._json({"started": True})
            return
        if u.path == "/api/cancel":
            with LOCK:
                if JOB["cur"] and JOB["cur"].status == "running":
                    JOB["cur"].stop_flag = True
            self._json({"ok": True})
            return
        self._json({"error": "not found"}, 404)


class Srv(ThreadingHTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description="model-bench web UI")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--bind", default="0.0.0.0")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    srv = Srv((a.bind, a.port), Handler)
    print("model-bench ui on http://%s:%d  (runs dir: %s)" % (a.bind, a.port, RUNS), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
