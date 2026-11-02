#!/usr/bin/env python3
"""Independent audit of saved GLM A/B/R requests, cache hits and NPU events."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
plan = json.loads((root / "plan.json").read_text())
summary = json.loads((root / "results/summary.json").read_text())
rows = [json.loads(s) for s in (root / "results/results.jsonl").read_text().splitlines()]
assert len(plan["questions"]) == len(rows) == 100
assert plan["num_hidden_layers"] == 78
assert plan["indexer_layer_indices"] == list(range(78))
assert len({r["qid"] for r in rows}) == 100
checks = {"questions": 0, "cold_a": 0, "pure_l2_r": 0,
          "eight_rank_d2h": 0, "eight_rank_h2d": 0,
          "a_correct": 0, "r_correct": 0, "answers_equal": 0,
          "tokens_equal": 0, "a_right_r_wrong": 0,
          "a_wrong_r_right": 0, "both_wrong_different": 0}
differences = []


def event_coverage(events, phase_id, direction):
    selected = [e for e in events if e.get("phase_id") == phase_id]
    assert not any(e.get("event") == "audit_error" for e in selected)
    selected = [e for e in selected if e.get("event") == "aiv_complete"
                and e.get("return_code") == 0 and e.get("direction") == direction
                and e.get("layout") == "layer_first_kv_split"
                and e.get("layers") == 78
                and e.get("indexer_layers") == list(range(78))
                and e.get("pages", 0) > 0 and e.get("bytes", 0) > 0]
    for rank in range(8):
        for component in ("k", "v", "index_k"):
            covered = set()
            for e in selected:
                if e.get("rank") != rank:
                    continue
                for part in e.get("components", []):
                    if part.get("name") != component:
                        continue
                    assert part.get("width_bytes", 0) > 0
                    lo, hi = part["lo"], part["hi"]
                    assert 0 <= lo < hi <= 78
                    covered.update(range(lo, hi))
            assert covered == set(range(78)), (rank, component, direction, len(covered))
    return len(selected)


for expected, row in zip(plan["questions"], rows):
    qid = expected["qid"]
    assert row["qid"] == qid
    case = root / "results" / qid
    assert json.loads((case / "verdict.json").read_text()) == row
    requests = [json.loads(s) for s in (case / "requests.jsonl").read_text().splitlines()]
    responses = [json.loads(s) for s in (case / "responses.jsonl").read_text().splitlines()]
    assert [x["phase"] for x in requests] == [x["phase"] for x in responses]
    assert requests[0]["phase"] == "A" and requests[-1]["phase"] == "R"
    assert all(x["phase"] == "B" for x in requests[1:-1])
    assert requests[0]["json"]["input_ids"] == requests[-1]["json"]["input_ids"] == expected["input_ids"]
    assert requests[0]["phase_id"] == responses[0]["phase_id"]
    assert requests[-1]["phase_id"] == responses[-1]["phase_id"]
    a = responses[0]["response"]
    r = responses[-1]["response"]
    assert a["meta_info"]["cached_tokens"] == 0
    checks["cold_a"] += 1
    tiers = r["meta_info"]["cached_tokens_details"]
    assert tiers["device"] == 0 and tiers["host"] >= 128
    checks["pure_l2_r"] += 1
    evidence = json.loads((case / "aiv_evidence.json").read_text())
    assert event_coverage(evidence, requests[0]["phase_id"], "D2H") > 0
    checks["eight_rank_d2h"] += 1
    assert event_coverage(evidence, requests[-1]["phase_id"], "H2D") > 0
    checks["eight_rank_h2d"] += 1
    assert row["token_match"] == (a["output_ids"] == r["output_ids"])
    for key, check in (("a_correct", "baseline_correct"), ("r_correct", "replay_correct"),
                       ("answers_equal", "answer_match"), ("tokens_equal", "token_match")):
        checks[key] += int(row[check])
    if not row["answer_match"]:
        differences.append({"qid": qid, "ground_truth": row["ground_truth"],
                            "a": row["baseline_answer"], "r": row["replay_answer"],
                            "a_correct": row["baseline_correct"], "r_correct": row["replay_correct"]})
        if row["baseline_correct"] and not row["replay_correct"]:
            checks["a_right_r_wrong"] += 1
        elif not row["baseline_correct"] and row["replay_correct"]:
            checks["a_wrong_r_right"] += 1
        else:
            checks["both_wrong_different"] += 1
    checks["questions"] += 1

assert summary["baseline_correct_completed"] == checks["a_correct"]
assert summary["replay_correct_completed"] == checks["r_correct"]
assert summary["answer_match_count"] == checks["answers_equal"]
assert summary["token_match_count"] == checks["tokens_equal"]
out = {"checks": checks, "differences": differences,
       "complete": True, "run_dir": str(root)}
(root / "independent_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
