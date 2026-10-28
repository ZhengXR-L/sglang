#!/usr/bin/env python3
"""Long-few-shot GSM8K: plan, launch audited server, then serial A/B/R eval."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

POOLS = ("swa", "deepseek_v4_c4", "deepseek_v4_c4_indexer",
         "deepseek_v4_c4_indexer_scale", "deepseek_v4_c4_state",
         "deepseek_v4_c4_indexer_state", "deepseek_v4_c128")


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_data(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get("question"), str) or not isinstance(r.get("answer"), str) for r in rows):
        raise ValueError(f"Expected GSM8K JSONL with question/answer strings: {path}")
    return rows


def answer(text):
    text = text.rsplit("####", 1)[-1].replace(",", "")
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    try:
        return str(Decimal(nums[-1]).normalize()) if nums else None
    except InvalidOperation:
        return None


def build_plan(train, test, tokenizer, minimum, maximum_shots, seed, offset, count, model):
    selected = test[offset:offset + count]
    if len(selected) != count or count <= 0:
        raise ValueError("Test file does not contain the requested fixed evaluation range")
    normalize = lambda s: " ".join(s.split())
    test_questions = {normalize(r["question"]) for r in selected}
    if len(test_questions) != count:
        raise ValueError("Duplicate evaluation questions")
    if test_questions & {normalize(r["question"]) for r in train}:
        raise ValueError("Training examples overlap evaluation questions; use independent train/test splits")
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    shots = []
    prompts = None
    for index in indices[:maximum_shots]:
        shots.append(index)
        prefix = "\n\n".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}" for i in shots)
        if len(tokenizer.encode(prefix)) < minimum:
            continue
        prompts = [prefix + "\n\nQuestion: " + r["question"] + "\nAnswer:" for r in selected]
        encoded = [tokenizer.encode(s) for s in prompts]
        if min(map(len, encoded)) >= minimum:
            break
    else:
        raise ValueError("Not enough training examples to reach the minimum prompt length")
    if prompts is None:
        raise ValueError("No prompts built")
    questions = []
    for n, (row, prompt, ids) in enumerate(zip(selected, prompts, encoded)):
        truth = answer(row["answer"])
        if truth is None:
            raise ValueError(f"Cannot extract ground truth at test index {offset+n}")
        questions.append({"qid": f"test-{offset+n}", "test_index": offset+n,
                          "question": row["question"], "ground_truth": truth,
                          "prompt": prompt, "input_ids": ids, "input_sha256": digest(ids)})
    return {"version": 1, "model_path": str(Path(model).resolve()), "seed": seed,
            "train_sha256": digest(train), "test_sha256": digest(test),
            "shot_indices": shots, "num_shots": len(shots), "test_offset": offset,
            "min_prompt_tokens": minimum, "raw_c128_group": 2048,
            "questions": questions}


class AuditReader:
    def __init__(self, root):
        self.root = Path(root)
        self.offsets = {}
        self.events = []

    def poll(self):
        for path in self.root.glob("calls-*.jsonl"):
            with path.open(encoding="utf-8") as src:
                src.seek(self.offsets.get(path, 0))
                while True:
                    pos = src.tell()
                    line = src.readline()
                    if not line or not line.endswith("\n"):
                        src.seek(pos)
                        break
                    self.events.append(json.loads(line))
                self.offsets[path] = src.tell()


def coverage(events, phase_id, direction, rank, pools=POOLS):
    rows = [e for e in events if e.get("phase_id") == phase_id and e.get("rank") == rank]
    errors = [e for e in rows if e.get("event") == "audit_error"]
    if errors:
        raise RuntimeError(f"AIV audit error: {errors[0]}")
    rows = [e for e in rows if e.get("event") == "aiv_complete" and e.get("direction") == direction
            and e.get("return_code") == 0 and e.get("pages", 0) > 0 and e.get("bytes", 0) > 0
            and e.get("layout") == "layer_first_kv_split"]
    result = {}
    for pool in pools:
        part = [e for e in rows if e.get("pool") == pool]
        counts = {e["layers"] for e in part}
        layers = sorted({e["layer"] for e in part})
        complete = len(counts) == 1 and next(iter(counts)) > 0 and layers == list(range(next(iter(counts))))
        result[pool] = {"complete": complete, "layers": layers,
                        "expected_layers": sorted(counts), "calls": len(part),
                        "bytes": sum(e["bytes"] for e in part)}
    return all(x["complete"] for x in result.values()), result


def wait_aiv(reader, phase_id, direction, rank, timeout):
    deadline = time.monotonic() + timeout
    while True:
        reader.poll()
        ok, detail = coverage(reader.events, phase_id, direction, rank)
        if ok:
            return detail
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Missing completed {direction} AIV coverage: {detail}")
        time.sleep(0.1)


def http(base, path, payload=None, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return raw.decode() if path == "/flush_cache" else json.loads(raw)


def preflight(info, plan, rank, eviction_tokens, max_new):
    if info.get("hicache_mem_layout") != "layer_first_kv_split" or info.get("hicache_io_backend") != "kernel_ascend":
        raise ValueError("Require resolved layer_first_kv_split + kernel_ascend")
    if info.get("hicache_write_policy") != "write_through":
        raise ValueError("Require write_through for the explicit A-stage D2H gate")
    if not info.get("enable_dp_attention") or not info.get("enable_dp_lm_head"):
        raise ValueError("Use the documented DP8 attention + DP lm-head configuration")
    dp = info.get("attn_dp_size") or info.get("dp_size")
    if dp == 1 and len(info.get("internal_states", [])) == info.get("tp_size"):
        dp = len(info["internal_states"])
    if dp != info.get("tp_size") or dp != 8 or info.get("pp_size", 1) != 1 or info.get("nnodes", 1) != 1:
        raise ValueError("This audit recipe supports single-host TP8/attention-DP8/PP1 only")
    if not 0 <= rank < dp:
        raise ValueError("Invalid routed DP rank")
    if info.get("speculative_algorithm") is not None:
        raise ValueError("This recipe requires MTP disabled")
    if 128 * (info.get("c128_page_size") or 16) != plan["raw_c128_group"]:
        raise ValueError("C128 group differs from the prepared plan")
    if Path(info["model_path"]).resolve() != Path(plan["model_path"]).resolve():
        raise ValueError("Plan tokenizer/model path differs from server model")
    caps = [s.get("memory_usage", {}).get("token_capacity") for s in info.get("internal_states", [])]
    caps = [c for c in caps if isinstance(c, int) and c > 0]
    if not caps:
        raise ValueError("Server does not expose actual per-worker token_capacity")
    capacity = min(caps)
    if eviction_tokens + 8 > capacity or max(len(q["input_ids"]) for q in plan["questions"]) + max_new > capacity:
        raise ValueError(f"Requests exceed actual capacity {capacity}")
    if min(len(q["input_ids"]) for q in plan["questions"]) < 2 * plan["raw_c128_group"]:
        raise ValueError("Prompts too short for this C128 transfer test")
    return capacity


def evaluate(args):
    from transformers import AutoTokenizer
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    questions = plan.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Plan must contain nonempty questions")
    qids = [q.get("qid", "") for q in questions]
    if len(set(qids)) != len(qids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", qid) for qid in qids):
        raise ValueError("Invalid or duplicate plan question IDs")
    if any(digest(q["input_ids"]) != q.get("input_sha256") for q in questions):
        raise ValueError("Plan input IDs failed SHA256 validation")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    audit = Path(args.audit_dir)
    if not (audit / "launch.json").exists():
        raise ValueError("Start the server with this script's serve command and the same audit-dir")
    lock = audit / "evaluation.lock"
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lock_fd, str(os.getpid()).encode()); os.close(lock_fd)
    results = []
    failure = None
    run_id = uuid.uuid4().hex
    reader = AuditReader(audit)
    try:
        info = http(args.base_url, "/server_info")
        dump(out / "server_info.json", info)
        capacity = preflight(info, plan, args.dp_rank, args.eviction_tokens, args.max_new_tokens)
        reader.poll()
        if not any(e.get("event") == "hook_ready" for e in reader.events):
            raise ValueError("No worker audit hook found; cannot claim per-question AIV coverage")
        tokenizer = AutoTokenizer.from_pretrained(plan["model_path"], trust_remote_code=True)
        rng = random.Random(args.eviction_seed)
        forbidden = set(tokenizer.all_special_ids) | {q["input_ids"][0] for q in plan["questions"]}
        b_ids = []
        special = set(tokenizer.all_special_ids)
        while len(b_ids) < args.eviction_tokens:
            token = rng.randrange(tokenizer.vocab_size)
            if token not in special and (b_ids or token not in forbidden):
                b_ids.append(token)
        dump(out / "eviction_input_ids.json", b_ids)
        dump(out / "arguments.json", {**vars(args), "run_id": run_id, "actual_capacity": capacity})

        def set_phase(qid, phase):
            phase_id = uuid.uuid4().hex
            dump(audit / "phase.json", {"run_id": run_id, "qid": qid, "phase": phase, "phase_id": phase_id})
            return phase_id

        def generate(case, qid, phase, ids):
            phase_id = set_phase(qid, phase)
            body = {"rid": phase_id, "input_ids": ids, "routed_dp_rank": args.dp_rank,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8 if phase == "B" else args.max_new_tokens,
                                        "ignore_eos": phase == "B", **({} if phase == "B" else {"stop": ["Question:"]})}}
            append(case / "requests.jsonl", {"phase": phase, "phase_id": phase_id, "json": body})
            result = http(args.base_url, "/generate", body, args.request_timeout)
            append(case / "responses.jsonl", {"phase": phase, "phase_id": phase_id, "response": result})
            if result.get("meta_info", {}).get("dp_rank") != args.dp_rank:
                raise ValueError("Server did not confirm the requested DP rank")
            return phase_id, result

        for q in plan["questions"]:
            case = out / q["qid"]; case.mkdir()
            set_phase(q["qid"], "FLUSH")
            http(args.base_url, "/flush_cache", {})
            phase_a, a = generate(case, q["qid"], "A", q["input_ids"])
            if a["meta_info"].get("cached_tokens") != 0:
                raise ValueError("A is not KV-cache cold; flush did not establish a valid baseline")
            d2h = wait_aiv(reader, phase_a, "D2H", args.dp_rank, args.audit_timeout)
            generate(case, q["qid"], "B", b_ids)
            phase_r, r = generate(case, q["qid"], "R", q["input_ids"])
            tiers = r["meta_info"].get("cached_tokens_details", {})
            if tiers.get("device") != 0 or tiers.get("host", 0) < plan["raw_c128_group"]:
                raise ValueError(f"R is not a proven pure L2 prefix hit: {tiers}")
            h2d = wait_aiv(reader, phase_r, "H2D", args.dp_rank, args.audit_timeout)
            dump(case / "aiv_evidence.json", [e for e in reader.events if e.get("phase_id") in (phase_a, phase_r)])
            pa, pr = answer(a.get("text", "")), answer(r.get("text", ""))
            ia, ir = a.get("output_ids", []), r.get("output_ids", [])
            valid = pa is not None and pr is not None and bool(ia) and bool(ir) and any(ia) and any(ir)
            row = {"qid": q["qid"], "ground_truth": q["ground_truth"], "baseline_answer": pa,
                   "replay_answer": pr, "baseline_correct": pa == q["ground_truth"],
                   "replay_correct": pr == q["ground_truth"], "answer_match": pa == pr,
                   "token_match": ia == ir, "valid": bool(valid), "path_covered": True,
                   "cached_tokens_details": tiers, "d2h": d2h, "h2d": h2d,
                   "passed": bool(valid and pa == pr and (not args.require_token_match or ia == ir))}
            results.append(row)
            dump(case / "verdict.json", row)
            append(out / "results.jsonl", row)
            print(json.dumps({k: row[k] for k in ("qid", "baseline_correct", "replay_correct", "answer_match", "token_match", "passed")}), flush=True)
    except Exception as exc:
        failure = repr(exc)
        dump(out / "error.json", {"error": failure, "completed_questions": len(results)})
    finally:
        reader.poll()
        dump(audit / "phase.json", {"run_id": run_id, "phase": "IDLE"})
        lock.unlink()
        total = len(plan["questions"])
        replay_accuracy = sum(r["replay_correct"] for r in results) / total
        complete = len(results) == total and failure is None
        absolute = None if args.min_accuracy is None or not complete else replay_accuracy >= args.min_accuracy
        summary = {"planned_questions": total, "covered_questions": len(results),
                   "path_coverage": len(results) / total, "all_questions_complete": complete,
                   "baseline_accuracy": sum(r["baseline_correct"] for r in results) / total if complete else None,
                   "replay_accuracy": replay_accuracy if complete else None,
                   "baseline_correct_completed": sum(r["baseline_correct"] for r in results),
                   "replay_correct_completed": sum(r["replay_correct"] for r in results),
                   "answer_match_count": sum(r["answer_match"] for r in results),
                   "token_match_count": sum(r["token_match"] for r in results),
                   "absolute_accuracy_gate": absolute, "error": failure,
                   "passed": complete and all(r["passed"] for r in results) and absolute is not False,
                   "protocol": "long-few-shot; relative answer preservation; not original 5-shot",
                   "cold_definition": "flush establishes KV-cache cold, not process/state-memory cold"}
        dump(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


def serve(args):
    if os.name != "posix":
        raise ValueError("serve must run inside the Linux NPU container")
    root = Path(args.audit_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if list(root.iterdir()):
        raise ValueError("Use a fresh empty audit-dir for each server start")
    boot = root / "boot"; boot.mkdir()
    shutil.copy2(Path(__file__).with_name("aiv_audit_hook.py"), boot / "aiv_audit_hook.py")
    (boot / "sitecustomize.py").write_text("from aiv_audit_hook import install\ninstall()\n")
    flags = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    if not flags:
        raise ValueError("Pass normal sglang.launch_server arguments after --")
    def flag_value(name, default):
        for i, flag in enumerate(flags):
            if flag == name:
                return flags[i+1]
            if flag.startswith(name + "="):
                return flag.split("=", 1)[1]
        return default
    with socket.socket() as sock:
        sock.bind((flag_value("--host", "127.0.0.1"), int(flag_value("--port", "30000"))))
    env = dict(os.environ)
    env["DSV4_GSM_AIV_AUDIT_DIR"] = str(root)
    env["SGLANG_HICACHE_HOST_MEM_BACKEND"] = "memfabric"
    env["PYTHONPATH"] = os.pathsep.join([str(boot), str(Path(args.sglang_root).resolve() / "python"), env.get("PYTHONPATH", "")])
    source_root = str(Path(args.sglang_root).resolve())
    commit = subprocess.check_output(["git", "-C", source_root, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", source_root, "status", "--short"], text=True)
    keys = ("ASCEND_RT_VISIBLE_DEVICES", "SGLANG_HICACHE_HOST_MEM_BACKEND", "SGLANG_ENABLE_WAR_BARRIER",
            "SGLANG_FORCE_COARSE_WAR_BARRIER", "SGLANG_NPU_USE_MULTI_STREAM", "HCCL_BUFFSIZE",
            "SGLANG_DSV4_FP4_EXPERTS", "SGLANG_OPT_USE_OVERLAP_STORE_CACHE")
    dump(root / "launch.json", {"argv": [sys.executable, "-m", "sglang.launch_server", *flags],
                               "sglang_root": source_root, "git_commit": commit, "git_status": status,
                               "environment": {k: env.get(k) for k in keys}, "audit_version": 1})
    os.execvpe(sys.executable, [sys.executable, "-m", "sglang.launch_server", *flags], env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Build immutable long-few-shot prompts; CPU only")
    plan.add_argument("--train", required=True); plan.add_argument("--test", required=True)
    plan.add_argument("--model-path", required=True); plan.add_argument("--output", required=True)
    plan.add_argument("--num-questions", type=int, default=100)
    plan.add_argument("--test-offset", type=int, default=0)
    plan.add_argument("--min-prompt-tokens", type=int, default=8192)
    plan.add_argument("--max-shots", type=int, default=128); plan.add_argument("--seed", type=int, default=0)
    server = sub.add_parser("serve", help="Launch an audited, dedicated SGLang server")
    server.add_argument("--audit-dir", required=True); server.add_argument("--sglang-root", required=True)
    server.add_argument("server_args", nargs=argparse.REMAINDER)
    ev = sub.add_parser("eval", help="Score serial A/D2H/B/R/H2D cases")
    ev.add_argument("--plan", required=True); ev.add_argument("--audit-dir", required=True)
    ev.add_argument("--base-url", default="http://127.0.0.1:32420"); ev.add_argument("--output", required=True)
    ev.add_argument("--dp-rank", type=int, default=0); ev.add_argument("--eviction-tokens", type=int, default=65536)
    ev.add_argument("--eviction-seed", type=int, default=60001); ev.add_argument("--max-new-tokens", type=int, default=512)
    ev.add_argument("--audit-timeout", type=float, default=180); ev.add_argument("--request-timeout", type=float, default=900)
    ev.add_argument("--require-token-match", action="store_true"); ev.add_argument("--min-accuracy", type=float)
    args = p.parse_args()
    if args.command == "plan":
        from transformers import AutoTokenizer
        if Path(args.output).exists():
            raise ValueError("Refusing to overwrite an existing plan")
        if args.min_prompt_tokens < 4096 or args.max_shots <= 0 or args.test_offset < 0:
            raise ValueError("Require min-prompt-tokens >=4096, positive max-shots and nonnegative test-offset")
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        value = build_plan(read_data(args.train), read_data(args.test), tok, args.min_prompt_tokens,
                           args.max_shots, args.seed, args.test_offset, args.num_questions, args.model_path)
        value.update(train_path=str(Path(args.train).resolve()), test_path=str(Path(args.test).resolve()))
        dump(args.output, value)
        print(json.dumps({"num_shots": value["num_shots"], "questions": len(value["questions"]),
                          "prompt_min": min(len(q["input_ids"]) for q in value["questions"]),
                          "prompt_max": max(len(q["input_ids"]) for q in value["questions"])}, indent=2))
        return 0
    if args.command == "serve":
        return serve(args)
    if args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1:
        raise ValueError("min-accuracy must be between 0 and 1")
    return evaluate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
