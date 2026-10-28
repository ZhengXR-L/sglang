#!/usr/bin/env python3
"""Cold/cold control for answer changes found in an A/B/R run."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
import uuid

from gsm8k_aiv import answer, http

source = Path(__file__).resolve().parent
prior = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
device = sys.argv[3]
root.mkdir(exist_ok=False)
scripts = root / "scripts"
scripts.mkdir()
for name in ("gsm8k_aiv.py", "aiv_audit_hook.py", "run_server.sh", "run_cold_repeat.py"):
    shutil.copy2(source / name, scripts / name)
plan = json.loads((prior / "plan.json").read_text())
rows = [json.loads(line) for line in (prior / "results/results.jsonl").read_text().splitlines()]
ids = {row["qid"] for row in rows if not row["answer_match"]}
cases = [q for q in plan["questions"] if q["qid"] in ids]
(root / "selected_qids.json").write_text(json.dumps([q["qid"] for q in cases], indent=2))
env = {**os.environ, "AUDIT_DIR": str(root / "audit"), "NPU_DEVICES": device, "PORT": "32345"}
server = None
results = []
error = None
try:
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(["bash", str(scripts / "run_server.sh")], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 900
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited: {server.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:32345/health", timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("Server health timeout")
        time.sleep(5)
    for q in cases:
        pair = []
        for repeat in range(2):
            flush_deadline = time.monotonic() + 120
            while True:
                try:
                    http("http://127.0.0.1:32345", "/flush_cache", {})
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 400 or time.monotonic() >= flush_deadline:
                        raise
                    time.sleep(0.2)
            request = {"rid": uuid.uuid4().hex, "input_ids": q["input_ids"],
                       "sampling_params": {"temperature": 0, "max_new_tokens": 512,
                                           "stop": ["Question:"]}}
            response = http("http://127.0.0.1:32345", "/generate", request, 900)
            pair.append({"request": request, "response": response,
                         "answer": answer(response.get("text", ""))})
        (root / f"{q['qid']}.json").write_text(json.dumps(pair, ensure_ascii=False, indent=2))
        old = next(x for x in rows if x["qid"] == q["qid"])
        result = {"qid": q["qid"], "ground_truth": q["ground_truth"],
                  "prior_a": old["baseline_answer"], "prior_r": old["replay_answer"],
                  "cold_1": pair[0]["answer"], "cold_2": pair[1]["answer"],
                  "cold_answer_match": pair[0]["answer"] == pair[1]["answer"],
                  "cold_token_match": pair[0]["response"].get("output_ids") == pair[1]["response"].get("output_ids"),
                  "both_cold": all(x["response"].get("meta_info", {}).get("cached_tokens") == 0 for x in pair)}
        results.append(result)
        print(json.dumps(result), flush=True)
except Exception as exc:
    error = repr(exc)
finally:
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)
    summary = {"selected": len(cases), "completed": len(results),
               "cold_answer_match_count": sum(x["cold_answer_match"] for x in results),
               "cold_token_match_count": sum(x["cold_token_match"] for x in results),
               "both_cold_count": sum(x["both_cold"] for x in results),
               "results": results, "error": error}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if error or len(results) != len(cases) or not all(x["both_cold"] for x in results):
    raise SystemExit(1)
