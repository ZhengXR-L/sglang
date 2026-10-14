#!/usr/bin/env python3
"""Probe an NPU HiCache server using only Python's standard library.

Start a dedicated server with max-running-requests=1 and debug logging, then:
  python3 scripts/verify_hicache_retry_interval_npu.py \
      --interval 8 --log /tmp/hicache-retry-i8.log

Exit codes: 0=PASS, 1=FAIL, 2=INCONCLUSIVE (insufficient retry evidence).
The server must include the retry observer logging in storage_prefetch.py.
"""

import argparse
import json
import re
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


EVENT = re.compile(
    r"HiCache storage retry event=(\w+) pid=(\d+) req=(\S+) "
    r"step=(\d+) interval=(-?\d+)(.*)"
)


def check(log_path, interval, offset=0):
    pending, failures, counts = {}, [], Counter()
    with open(log_path, "rb") as log:
        log.seek(offset)
        for line in log:
            match = EVENT.search(line.decode("utf-8", errors="replace"))
            if not match:
                continue
            kind, pid, rid, step, configured, tail = match.groups()
            step, configured = int(step), int(configured)
            fields = dict(re.findall(r"(\w+)=([^\s]+)", tail))
            key = (pid, rid)
            counts[kind] += 1
            if configured != interval:
                failures.append(f"{key}: logged interval={configured}, expected={interval}")
                continue
            if kind == "scheduled":
                due = int(fields["due_step"])
                if interval == 0 or due - step != interval:
                    failures.append(f"{key}: invalid scheduled step={step}, due={due}")
                pending[key] = (step, due)
            elif kind == "fired" and fields.get("mode") == "poll":
                scheduled = pending.pop(key, None)
                if interval == 0 or scheduled is None:
                    failures.append(f"{key}: disabled or unscheduled poll fired")
                elif step != scheduled[1] or int(fields["due_step"]) != scheduled[1]:
                    failures.append(f"{key}: fired step={step}, expected={scheduled[1]}")
                else:
                    counts["verified"] += 1
                    print(f"verified pid={pid} req={rid}: scheduled={scheduled[0]}, "
                          f"fired={step}, delta={step - scheduled[0]}")
            elif kind == "cancelled":
                pending.pop(key, None)
                if fields.get("reason") == "disabled":
                    counts["disabled"] += 1
                    if interval > 0:
                        failures.append(f"{key}: polling disabled with interval={interval}")
    print(f"summary: {dict(counts)}, pending={len(pending)}")
    if failures:
        print("FAIL:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    if (interval == 0 and counts["disabled"]) or (interval > 0 and counts["verified"]):
        print(f"PASS: interval={interval} " +
              ("disables miss polling" if interval == 0 else "matches actual retry step delta"))
        return 0
    print("INCONCLUSIVE: no qualifying queued-miss evidence. Check debug logging, "
          "L3 configuration, queue-head cancellations, and request duration.", file=sys.stderr)
    return 2


def generate(base_url, prompt, tokens, timeout):
    request = Request(
        base_url + "/generate",
        data=json.dumps({
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    meta = result.get("meta_info", {})
    if meta.get("finish_reason", {}).get("type") == "abort":
        raise RuntimeError(f"Request aborted: {meta}")
    print(f"request completed: tokens={meta.get('completion_tokens')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--holder-tokens", type=int, default=1024)
    parser.add_argument(
        "--queued-requests", type=int, default=6,
        help="Waiting requests (default: 6, targeting 5 retrying requests plus the queue head)",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--check-only", action="store_true", help="Check an existing log without sending requests")
    args = parser.parse_args()
    if args.interval < 0 or args.holder_tokens < 1 or args.queued_requests < 2 or args.timeout <= 0:
        parser.error("Require interval>=0, holder-tokens>=1, queued-requests>=2, timeout>0")
    if not args.log.is_file():
        parser.error(f"Log does not exist: {args.log}")
    if args.check_only:
        return check(args.log, args.interval)
    base_url = f"http://127.0.0.1:{args.port}"
    with urlopen(base_url + "/health", timeout=5):
        pass
    with args.log.open("rb") as log:
        if not any(b"HiCache storage retry observer initialized" in line for line in log):
            parser.error("Missing retry observer logging; use this workspace's storage_prefetch.py and restart the server")
    offset = args.log.stat().st_size
    run_id = uuid.uuid4().hex
    print(f"Reading new log bytes only: offset={offset}, run={run_id}", flush=True)
    # Distinct long prefixes trigger L3 misses even when rerunning the probe.
    # More than one waiting request is required: queue-head misses are cancelled.
    with ThreadPoolExecutor(max_workers=args.queued_requests + 1) as pool:
        futures = [pool.submit(generate, base_url,
            f"{run_id} holder: " + "Explain memory hierarchy in detail. " * 100,
            args.holder_tokens, args.timeout)]
        time.sleep(0.5)
        for index in range(args.queued_requests):
            futures.append(pool.submit(generate, base_url,
                f"{run_id} queued-{index}: " + "Describe storage prefetch scheduling in detail. " * 100,
                32, args.timeout))
        for future in as_completed(futures):
            future.result()
    time.sleep(1)
    return check(args.log, args.interval, offset)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
