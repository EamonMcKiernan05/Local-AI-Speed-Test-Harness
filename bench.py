#!/usr/bin/env python3
"""model-bench: fixed-battery prefill/decode speed harness for local LLM endpoints.

One battery, every time: exact-token prompt slices at standard depths, one request
per arm, prefill + decode tok/s read from the server's own `timings` object.
Never starts, stops or restarts a model server; it only measures what serves now.

Backends (auto-detected, overridable):
  llama.cpp   POST /completion            (ignore_eos, cache_prompt off, fixed seed/temp)
  Strata      POST /v1/chat/completions   (baseline sampler, timings + MTP draft counters)
  OpenAI-ish  POST /v1/chat/completions   (timings if the server reports them, else client SSE timing)

CLI:
  python3 bench.py --url http://127.0.0.1:8080 --label "Strata base IQ3_S"
  python3 bench.py --url http://127.0.0.1:8080 --battery quick --gen 256 --reps 2
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "0.1.0"
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- battery defaults
DEPTHS_STANDARD = [20000, 60000, 100000, 150000, 200000, 250000]
DEPTHS_QUICK = [20000, 150000]
BOOK_BY_DEPTH = {20000: "frankenstein", 60000: "sherlock", 100000: "pride",
                 150000: "tale2cities", 200000: "greatexp", 250000: "mobydick"}
BOOK_ORDER = ["frankenstein", "sherlock", "pride", "tale2cities", "greatexp", "mobydick"]
INSTR = ("[%s] Write a detailed technical explanation of lighthouse optics for an engineering audience. "
         "Aim for at least 400 words. ")
GEN_DEFAULT = 400
TEMP = 0.5
SEED = 12345
CHARS_PER_TOKEN = 3.6      # fallback only; replaced by measurement wherever a tokenizer is available
CTX_SAFETY = 512           # a prompt must fit ctx - gen - safety

STRATA_TOOLS = os.environ.get("MODEL_BENCH_STRATA_TOOLS", "/home/eamon/Strata/tools")
GGUF_CANDIDATES = [c for c in (os.environ.get("MODEL_BENCH_GGUF") or "").split(":") if c] or [
    "/mnt/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S ISTA-DASLab/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "/mnt/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Coder ISTA-DASLab/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Coder-00001-of-00002.gguf",
]
PACK_CANDIDATES = [c for c in (os.environ.get("MODEL_BENCH_PACK") or "").split(":") if c] or [
    "/mnt/models/Strata-data/packs/iq3_s/tokenizer",
    "/mnt/models/Strata-data/packs/coder-iq1_m/tokenizer",
]


def iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def r1(v):
    return round(v, 1) if isinstance(v, (int, float)) else None


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:44] or "run"


def fmt_depth(d):
    return ("%gk" % (d / 1000.0)) if d >= 1000 else str(d)


def parse_depths(s):
    """'20k,100k,5000' -> [20000, 100000, 5000]"""
    out = []
    for part in re.split(r"[,\s]+", (s or "").strip()):
        if not part:
            continue
        m = re.match(r"^(\d+)(k?)$", part.lower())
        if not m:
            raise ValueError("bad depth %r (use e.g. 20k, 100k, 5000)" % part)
        out.append(int(m.group(1)) * (1000 if m.group(2) == "k" else 1))
    return sorted(set(out))


def normalize_base(url):
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u


# ---------------------------------------------------------------- http
def http_json(url, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            excerpt = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            excerpt = ""
        raise RuntimeError("HTTP %s from %s: %s" % (e.code, url, excerpt)) from None


def http_json_soft(url, payload=None, timeout=8):
    try:
        return http_json(url, payload, timeout), None
    except Exception as e:
        return None, str(e)[:200]


def detect(base):
    """Probe an endpoint: backend, model id, context size. Works for llama.cpp, Strata, generic OpenAI."""
    base = normalize_base(base)
    info = {"endpoint": base, "backend": None, "model": None, "ctx": None}
    props, props_err = http_json_soft(base + "/props")
    models, models_err = http_json_soft(base + "/v1/models")
    if props is None and models is None:
        raise RuntimeError("no answer from %s (/props: %s; /v1/models: %s)"
                           % (base, props_err, models_err))

    ids, strata_marker, llama_marker = [], False, False
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        for d in models["data"]:
            if not isinstance(d, dict):
                continue
            if d.get("id"):
                ids.append(d["id"])
            # Strata's /v1/models items carry `architecture` + `status`; llama.cpp's do not.
            if "architecture" in d or "status" in d:
                strata_marker = True
            if d.get("owned_by") == "llamacpp":
                llama_marker = True
    if isinstance(props, dict):
        # llama.cpp /props carries these keys; Strata's llama.cpp-style /props does not.
        if any(k in props for k in ("chat_template_caps", "model_ftype", "endpoint_props", "build_info")):
            llama_marker = True

    if strata_marker:
        info["backend"] = "strata"
    elif llama_marker or isinstance(props, dict):
        info["backend"] = "llamacpp"
    else:
        info["backend"] = "openai"

    if isinstance(props, dict):
        dgs = props.get("default_generation_settings")
        if isinstance(dgs, dict):
            info["ctx"] = dgs.get("n_ctx")
    if ids:
        info["model"] = ids[0]
    if not info["model"] and isinstance(props, dict):
        info["model"] = props.get("model_alias") or props.get("model_path")
    if info["ctx"] is None and isinstance(models, dict):
        for d in (models.get("data") or []):
            if isinstance(d, dict) and isinstance(d.get("meta"), dict) and d["meta"].get("n_ctx"):
                info["ctx"] = d["meta"]["n_ctx"]
                break
    return info


# ---------------------------------------------------------------- tokenizer / prompt slicing
def load_strata_tokenizer():
    """Load the pack tokenizer for exact slicing (needs the Strata venv on .5 for the `regex` module)."""
    try:
        if STRATA_TOOLS not in sys.path:
            sys.path.insert(0, STRATA_TOOLS)
        import strata_tokenizer as ST
    except Exception as e:
        return None, "no strata_tokenizer (%s)" % str(e)[:80]
    last = "no vocab source"
    for p in GGUF_CANDIDATES:
        try:
            if Path(p).exists():
                return ST.Tokenizer.from_gguf(p), "gguf %s" % Path(p).parent.name[:40]
        except Exception as e:
            last = str(e)[:80]
    for pk in PACK_CANDIDATES:
        try:
            vocab = json.loads((Path(pk) / "vocab.json").read_text())
            tokens = [None] * len(vocab)
            for t, i in vocab.items():
                tokens[i] = t
            merges = (Path(pk) / "merges.txt").read_text().split("\n")
            types = json.loads((Path(pk) / "token_type.json").read_text())
            return ST.Tokenizer(tokens, merges, types), "pack %s" % Path(pk).parts[-2]
        except Exception as e:
            last = str(e)[:80]
    return None, last


class Sizer:
    """Slices corpus text to ~exact token targets, method recorded per run."""

    def __init__(self, base, backend):
        self.base, self.backend = base, backend
        self.tok, self.tok_note = load_strata_tokenizer()
        self._texts, self._ids = {}, {}
        self.calib = CHARS_PER_TOKEN
        self.method = "ratio estimate"

    def book_text(self, book):
        if book not in self._texts:
            p = HERE / "corpus" / (book + ".txt")
            if not p.exists():
                raise RuntimeError("corpus file missing: %s" % p)
            raw = p.read_text(encoding="utf-8", errors="replace")
            marker = "*** START OF THE PROJECT GUTENBERG"
            i = raw.find(marker)
            if i >= 0:
                j = raw.find("\n", i)
                raw = raw[j + 1:]
            self._texts[book] = raw
        return self._texts[book]

    def slice(self, book, target):
        """-> (text, token_count_or_None, method)"""
        text = self.book_text(book)
        if self.tok is not None:
            ids = self._ids.get(book)
            if ids is None:
                ids = self.tok.encode(text)
                self._ids[book] = ids
            n = min(int(target), len(ids))
            out = self.tok.decode(ids[:n])
            c = len(self.tok.encode(out))
            for _ in range(12):
                if c == target:
                    break
                n = max(1, min(len(ids), n + (1 if c < target else -1)))
                out = self.tok.decode(ids[:n])
                c = len(self.tok.encode(out))
            self.method = "local tokenizer (exact, %s)" % self.tok_note
            return out, c, self.method

        if self.backend == "llamacpp":
            n = min(len(text), max(1, int(int(target) * self.calib)))
            best, c = text[:n], None
            for _ in range(10):
                r = http_json(self.base + "/tokenize", {"content": best}, timeout=120)
                c = len(r.get("tokens") or [])
                if c == int(target) or c <= 0:
                    break
                n = max(1, min(len(text), n + round((int(target) - c) * (n / c))))
                best = text[:n]
            if c:
                self.calib = len(best) / max(1, int(target))
            self.method = "server /tokenize"
            return best, c, self.method

        n = min(len(text), max(1, int(int(target) * self.calib)))
        return text[:n], None, "ratio estimate"


# ---------------------------------------------------------------- request adapters
def request_llamacpp(base, text, gen, timeout):
    d = http_json(base + "/completion",
                  {"prompt": text, "n_predict": gen, "cache_prompt": False,
                   "seed": SEED, "temperature": TEMP, "ignore_eos": True},
                  timeout=timeout)
    tt = d.get("timings") or {}
    return {"ptok": tt.get("prompt_n"), "gen": tt.get("predicted_n"),
            "prefill_tps": r1(tt.get("prompt_per_second")),
            "decode_tps": r1(tt.get("predicted_per_second")),
            "draft_n": tt.get("draft_n"), "draft_acc": tt.get("draft_n_accepted"),
            "timings_source": "server"}


def request_openai_stream(base, model, text, gen, timeout):
    """Fallback for servers without a timings object: client-side SSE timing (approximate)."""
    body = {"model": model or "x", "messages": [{"role": "user", "content": text}],
            "max_tokens": gen, "temperature": TEMP, "seed": SEED, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    t0, ttft, last, n = time.time(), None, None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            for ch in (obj.get("choices") or []):
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.time()
                    if ttft is None:
                        ttft = now
                    n += 1
                    last = now
    dec = (n - 1) / (last - ttft) if (n > 1 and ttft and last and last > ttft) else None
    return {"ptok": None, "gen": n, "prefill_tps": None,
            "decode_tps": r1(dec), "draft_n": None, "draft_acc": None,
            "timings_source": "client stream (approx)"}


def request_openai(base, model, text, gen, timeout):
    d = http_json(base + "/v1/chat/completions",
                  {"model": model or "x", "messages": [{"role": "user", "content": text}],
                   "max_tokens": gen, "temperature": TEMP, "seed": SEED},
                  timeout=timeout)
    tt = d.get("timings") or {}
    u = d.get("usage") or {}
    if tt:
        return {"ptok": u.get("prompt_tokens"), "gen": u.get("completion_tokens"),
                "prefill_tps": r1(tt.get("prompt_per_second")),
                "decode_tps": r1(tt.get("predicted_per_second")),
                "draft_n": tt.get("draft_n"), "draft_acc": tt.get("draft_n_accepted"),
                "timings_source": "server"}
    return request_openai_stream(base, model, text, gen, timeout)


def request_strata(base, model, text, gen, timeout):
    """Strata: keep the sweep baseline sampler (top_k 20 / top_p 0.95 / min_p 0.0), timings expected."""
    d = http_json(base + "/v1/chat/completions",
                  {"model": model or "x", "messages": [{"role": "user", "content": text}],
                   "max_tokens": gen, "temperature": TEMP, "seed": SEED,
                   "top_k": 20, "top_p": 0.95, "min_p": 0.0},
                  timeout=timeout)
    tt = d.get("timings") or {}
    u = d.get("usage") or {}
    if tt:
        return {"ptok": u.get("prompt_tokens"), "gen": u.get("completion_tokens"),
                "prefill_tps": r1(tt.get("prompt_per_second")),
                "decode_tps": r1(tt.get("predicted_per_second")),
                "draft_n": tt.get("draft_n"), "draft_acc": tt.get("draft_n_accepted"),
                "timings_source": "server"}
    return request_openai_stream(base, model, text, gen, timeout)


def dispatch(base, backend, model, prompt, gen, timeout):
    if backend == "llamacpp":
        try:
            return request_llamacpp(base, prompt, gen, timeout)
        except RuntimeError as e:
            if "HTTP 404" in str(e) or "HTTP 405" in str(e):
                r = request_openai(base, model, prompt, gen, timeout)
                r["note"] = "llama.cpp /completion unavailable; fell back to /v1/chat/completions"
                return r
            raise
    if backend == "strata":
        return request_strata(base, model, prompt, gen, timeout)
    return request_openai(base, model, prompt, gen, timeout)


# ---------------------------------------------------------------- system info
def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=index,name,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        gpus = []
        for line in out.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 5:
                gpus.append({"idx": int(p[0]), "name": p[1], "used_mib": int(p[2]),
                             "total_mib": int(p[3]), "temp_c": int(p[4])})
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------- runner
def run_job(cfg, on_row=None, should_stop=None, log=None):
    """Run one battery. cfg: url, label, model, backend, depths, gen, reps, timeout, out_root."""
    log = log or (lambda s: None)
    base = normalize_base(cfg.get("url") or "")
    if not base:
        raise RuntimeError("no endpoint url given")
    det = detect(base)
    backend = cfg.get("backend") or det["backend"]
    if cfg.get("model"):
        det["model"] = cfg["model"]
    gen = max(16, int(cfg.get("gen") or GEN_DEFAULT))
    reps = max(1, int(cfg.get("reps") or 1))
    timeout = max(60, int(cfg.get("timeout") or 2400))
    depths = sorted(set(int(d) for d in (cfg.get("depths") or DEPTHS_STANDARD) if int(d) > 0))
    out_root = Path(cfg.get("out_root") or (HERE / "runs"))
    label = (cfg.get("label") or "").strip() or (det.get("model") or "run")

    ctx = det.get("ctx")
    skipped = []
    if ctx:
        keep = []
        for d in depths:
            if d + gen + CTX_SAFETY > int(ctx):
                skipped.append({"depth_k": d,
                                "reason": "needs ctx >= %d, server has %d" % (d + gen + CTX_SAFETY, ctx)})
            else:
                keep.append(d)
        depths = keep
    if not depths and not skipped:
        raise RuntimeError("no depths to run")

    run_id = time.strftime("%Y-%m-%dT%H%MZ", time.gmtime()) + "-" + slug(label)
    rd = out_root / run_id
    k = 2
    while rd.exists():
        rd = Path(str(out_root / run_id) + "-%d" % k)
        k += 1
    rd.mkdir(parents=True, exist_ok=True)
    jsonl = rd / "results.jsonl"

    meta = {"run_id": rd.name, "tool": "model-bench " + VERSION, "label": label,
            "endpoint": base, "backend": backend, "detected": det,
            "battery": {"depths_k": depths, "gen": gen, "reps": reps, "temp": TEMP, "seed": SEED,
                        "instruction": INSTR % "{tag}", "skipped": skipped},
            "host": {"hostname": os.uname().nodename, "gpus": gpu_snapshot()},
            "started_utc": iso(), "finished_utc": None}
    (rd / "meta.json").write_text(json.dumps(meta, indent=2))

    sizer = Sizer(base, backend)

    def emit(row):
        row = dict(row)
        row["ts"] = iso()
        row["endpoint"] = base
        row["backend"] = backend
        row["label"] = label
        row["run_id"] = rd.name
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        if on_row:
            try:
                on_row(row)
            except Exception:
                pass
        if row.get("kind") == "arm":
            log("arm %s rep %s/%s: prefill %s tok/s, decode %s tok/s, %s gen, %ss%s"
                % (fmt_depth(row["depth_k"]), row.get("rep"), reps,
                   row.get("prefill_tps"), row.get("decode_tps"), row.get("gen"),
                   row.get("wall_s"), (" ERROR: " + row["error"][:80]) if row.get("error") else ""))

    for s in skipped:
        emit({"kind": "skipped", "depth_k": s["depth_k"], "reason": s["reason"]})

    stopped = False
    try:
        if not (should_stop and should_stop()):
            wtext, wc, wm = sizer.slice(BOOK_ORDER[0], 1024)
            t0 = time.time()
            try:
                r = dispatch(base, backend, det.get("model"), (INSTR % "warmup") + wtext, 16, timeout)
                emit({"kind": "warmup", "depth_k": 1024, "book": BOOK_ORDER[0], "method": wm,
                      "sliced_tokens": wc, "wall_s": round(time.time() - t0, 1),
                      "gpus": gpu_snapshot(), **r})
            except Exception as e:
                emit({"kind": "warmup", "depth_k": 1024, "book": BOOK_ORDER[0], "method": wm,
                      "error": str(e)[:300], "wall_s": round(time.time() - t0, 1)})

        for di, depth in enumerate(depths):
            book = BOOK_BY_DEPTH.get(depth) or BOOK_ORDER[di % len(BOOK_ORDER)]
            for rep in range(1, reps + 1):
                if should_stop and should_stop():
                    stopped = True
                    break
                text, sc, method = sizer.slice(book, depth)
                tag = "r%06d" % random.randint(0, 999999)
                prompt = (INSTR % tag) + text
                t0 = time.time()
                row = {"kind": "arm", "depth_k": depth, "book": book, "rep": rep,
                       "method": method, "sliced_tokens": sc}
                try:
                    row.update(dispatch(base, backend, det.get("model"), prompt, gen, timeout))
                except Exception as e:
                    row["error"] = str(e)[:300]
                row["wall_s"] = round(time.time() - t0, 1)
                row["gpus"] = gpu_snapshot()
                emit(row)
            if stopped:
                break
        if should_stop and should_stop():
            stopped = True
    finally:
        fin = summarize(rd)
        meta["finished_utc"] = iso()
        meta["rows"] = fin["rows"]
        meta["error_count"] = len(fin["errors"])
        meta["aborted"] = bool(stopped)
        (rd / "meta.json").write_text(json.dumps(meta, indent=2))
    return rd


def summarize(rd):
    """Fold results.jsonl into summary.md + summary.csv. Returns {rows, errors, means}."""
    rd = Path(rd)
    rows = []
    for line in (rd / "results.jsonl").read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").exists() else {}
    arms = [r for r in rows if r.get("kind") == "arm"]
    ok = [r for r in arms if r.get("decode_tps") is not None and not r.get("error")]
    errors = [r for r in rows if r.get("error")]

    by_depth = {}
    for r in ok:
        by_depth.setdefault(r["depth_k"], []).append(r)
    means = []
    for d in sorted(by_depth):
        rs = by_depth[d]
        means.append({
            "depth_k": d,
            "ptok": mean([r.get("ptok") for r in rs]),
            "prefill_tps": mean([r.get("prefill_tps") for r in rs]),
            "decode_tps": mean([r.get("decode_tps") for r in rs]),
            "gen": mean([r.get("gen") for r in rs]),
            "wall_s": mean([r.get("wall_s") for r in rs]),
            "reps": len(rs),
        })

    det = meta.get("detected") or {}
    bat = meta.get("battery") or {}
    method = ok[0].get("method") if ok else (arms[0].get("method") if arms else "n/a")
    lines = ["# Model Bench: %s" % meta.get("label", ""), "",
             "- endpoint: `%s` (%s)" % (meta.get("endpoint"), meta.get("backend")),
             "- model: `%s`%s" % (det.get("model"), (" · ctx %s" % det.get("ctx")) if det.get("ctx") else ""),
             "- battery: depths %s · gen %s · reps %s · temp %s · seed %s"
             % (", ".join(fmt_depth(d) for d in bat.get("depths_k", [])), bat.get("gen"),
                bat.get("reps"), bat.get("temp"), bat.get("seed")),
             "- prompts: %s" % method,
             "- run: %s · %s UTC" % (rd.name, meta.get("started_utc", "")),
             ""]
    if means:
        lines += ["| depth | prompt tok | prefill tok/s | decode tok/s | gen tok | wall s | reps |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m in means:
            lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                         % (fmt_depth(m["depth_k"]), int(m["ptok"]) if m["ptok"] else "",
                            f1(m["prefill_tps"]), f1(m["decode_tps"]),
                            int(m["gen"]) if m["gen"] else "", f1(m["wall_s"]), m["reps"]))
        pk = max(means, key=lambda m: m["prefill_tps"] or 0)
        dk = max(means, key=lambda m: m["decode_tps"] or 0)
        lines += ["",
                  "- prefill peak: **%s tok/s** at %s" % (f1(pk["prefill_tps"]), fmt_depth(pk["depth_k"])),
                  "- decode peak: **%s tok/s** at %s" % (f1(dk["decode_tps"]), fmt_depth(dk["depth_k"]))]
    for s in (bat.get("skipped") or []):
        lines.append("- skipped %s: %s" % (fmt_depth(s["depth_k"]), s["reason"]))
    if errors:
        lines.append("")
        lines.append("## Errors")
        for e in errors:
            lines.append("- %s%s: %s" % (e.get("kind"), (" " + fmt_depth(e["depth_k"])) if e.get("depth_k") else "",
                                         e.get("error", "")[:200]))
    (rd / "summary.md").write_text("\n".join(lines) + "\n")

    with open(rd / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "depth_k", "rep", "book", "method", "sliced_tokens", "ptok", "gen",
                    "prefill_tps", "decode_tps", "draft_n", "draft_acc", "wall_s",
                    "gpu0_used_mib", "gpu0_temp_c", "gpu1_used_mib", "gpu1_temp_c", "error"])
        for r in arms:
            g = r.get("gpus") or []
            g0 = g[0] if len(g) > 0 else {}
            g1 = g[1] if len(g) > 1 else {}
            w.writerow([r.get("run_id"), r.get("depth_k"), r.get("rep"), r.get("book"), r.get("method"),
                        r.get("sliced_tokens"), r.get("ptok"), r.get("gen"), r.get("prefill_tps"),
                        r.get("decode_tps"), r.get("draft_n"), r.get("draft_acc"), r.get("wall_s"),
                        g0.get("used_mib"), g0.get("temp_c"), g1.get("used_mib"), g1.get("temp_c"),
                        r.get("error") or ""])
    return {"rows": len(rows), "errors": errors, "means": means}


def mean(vals):
    vs = [v for v in vals if isinstance(v, (int, float))]
    return sum(vs) / len(vs) if vs else None


def f1(v):
    return ("%.1f" % v) if isinstance(v, (int, float)) else ""


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="model-bench: fixed prefill/decode battery")
    ap.add_argument("--url", required=True, help="endpoint base, e.g. http://127.0.0.1:8080")
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--backend", default="", choices=["", "llamacpp", "strata", "openai"])
    ap.add_argument("--battery", default="standard", choices=["standard", "quick"])
    ap.add_argument("--depths", default="", help="custom depths, e.g. 20k,100k,5000")
    ap.add_argument("--gen", type=int, default=GEN_DEFAULT)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    try:
        depths = parse_depths(a.depths) if a.depths else (
            DEPTHS_QUICK if a.battery == "quick" else DEPTHS_STANDARD)
    except ValueError as e:
        raise SystemExit(str(e))
    cfg = {"url": a.url, "label": a.label, "model": a.model, "backend": a.backend,
           "depths": depths, "gen": a.gen, "reps": a.reps, "timeout": a.timeout}
    if a.out:
        cfg["out_root"] = a.out
    print("model-bench %s -> %s" % (VERSION, a.url), flush=True)
    rd = run_job(cfg, log=lambda s: print(s, flush=True))
    print("\n" + (rd / "summary.md").read_text(), flush=True)
    print("run dir: %s" % rd, flush=True)


if __name__ == "__main__":
    main()
