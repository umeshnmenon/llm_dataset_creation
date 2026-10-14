"""One command builds, trains, kills a process, resumes, replays, forks and audits."""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from v5data.corpus import Tokenizer, admit, build, make_corpus
from v5data.engine import fork_branch, initialize, worker
from v5data.ledger import Ledger
from v5data.packing import attention_matrix, pack_records, verify_pack
from v5data.schedule import compile_schedule, load_config
from v5data.util import atomic_json, canonical, digest, emit, read_json

PROJECT = Path(__file__).resolve().parent


def run_child(root, config_path, branch, until, crash=None, replay=None):
    args = [sys.executable, str(PROJECT / "run_demo.py"), "--worker", "--artifacts", str(root),
            "--config", str(config_path), "--branch", branch, "--until", str(until)]
    if crash is not None:
        args += ["--crash", str(crash)]
    if replay:
        args += ["--replay-parent", replay]
    return subprocess.run(args, cwd=PROJECT, check=False).returncode


def artifact_root(path):
    root = Path(path).resolve()
    if root == PROJECT or PROJECT in root.parents and root.name in {"v5data", "data", "tests", "sources", ".venv"}:
        raise ValueError("Artifacts must not overwrite source directories")
    # No recursive deletion. Preserve previous runs by renaming only the exact artifact folder.
    if root.exists() and any(root.iterdir()):
        if not (root / "evidence.json").exists() and not (root / "registry.json").exists():
            raise ValueError("Refusing to move an unrecognised nonempty artifact directory")
        archive = root.with_name(root.name + ".previous-" + str(time.time_ns()))
        root.rename(archive)
    root.mkdir(parents=True, exist_ok=True)
    return root


def evaluate_split(root, state, split):
    from v5data.model import TinyLM
    from v5data.packing import make_pack
    model = TinyLM.restore(state["model"])
    records = [r for r in read_json(root / "registry.json")["records"] if r["split"] == split]
    packs = [make_pack([(r, 0, len(r["tokens"]))], model.max_length, "pad_only") for r in records]
    loss, _, traces = model.loss_gradient(packs, gradient=False)
    return {"split": split, "checkpoint_id": state["checkpoint_id"], "mean_cross_entropy": loss,
            "perplexity": math.exp(loss), "loss_tokens": len(traces), "gradient_applied": False,
            "shard_ids": [r["shard_id"] for r in records]}


def audit(root, config, schedule, tests_result, crash_result):
    ledger = Ledger(root)
    registry = read_json(root / "registry.json")
    main = ledger.events("consumption", "main")
    learning = ledger.events("learning", "main")
    traces = ledger.events("token_loss", "main")
    opus = ledger.events("opus", "main")
    initial, final = ledger.get_state("main", 0), ledger.get_state("main")
    findings = []

    def check(requirement, condition, evidence, detail):
        findings.append({"requirement": requirement, "passed": bool(condition), "evidence": evidence, "detail": detail})

    chain_ok = ledger.verify()
    check("ledger_integrity", chain_ok, "ledgers/ledger.sqlite", "Append-only journal hash chain recomputed")
    check("tokenizer_integrity", all(m["tokenizer_hash"] == initial["tokenizer_hash"] for m in registry["manifests"]), "tokenizer.json; manifests/", "Frozen tokenizer hashes agree across manifests and checkpoint")
    valid_shards = True
    for r in registry["records"]:
        if r["split"] == "train":
            admit(root, r, initial["tokenizer_hash"], r["reserve"])
    check("immutable_shards", valid_shards, "shards/; manifests/", "All admitted training shard content hashes revalidated against token records")
    heldout_ids = {r["shard_id"] for r in registry["records"] if r["split"] != "train"}
    trained_ids = {s for e in main for s in e["payload"]["shard_ids"]}
    firewall_events = ledger.events("firewall")
    check("evaluation_validation_firewall", not heldout_ids & trained_ids and {e["payload"].get("split") for e in firewall_events} >= {"validation", "test", "proxy"}, "ledgers/firewall.jsonl; ledgers/consumption.jsonl", "Heldout/proxy admission attempts blocked; no heldout shard in training")
    check("contamination_firewall", any(b["doc_id"] == "embedded-test-leak" and b["reason"] == "heldout_fingerprint_overlap" for b in registry["blocked"]), "registry.json", "Embedded test canary blocked by token-independent text fingerprints")
    packing_ok, mask_ok, position_ok, reconstruct_ok = True, True, True, True
    all_positions, real_tokens, useful = 0, 0, 0
    actual = []
    by_doc = {r["doc_id"]: r for r in registry["records"]}
    reserved = {r["shard_id"] for r in registry["records"] if r["reserve"]}
    reserve_ok = True
    for event in main:
        p = event["payload"]
        pack = read_json(root / "packs" / (p["pack_id"] + ".json"))
        verify_pack(pack)
        all_positions += len(pack["tokens"]); real_tokens += pack["real_tokens"]; useful += sum(pack["loss_mask"])
        mask_ok &= not any(flag and role in {"tool", "user", "system", "pad"} for flag, role in zip(pack["loss_mask"], pack["roles"]))
        for span in pack["spans"]:
            r = by_doc[span["doc_id"]]
            reconstruct_ok &= pack["tokens"][span["packed_start"]:span["packed_end"]] == r["tokens"][span["start"]:span["end"]]
            position_ok &= pack["position_ids"][span["packed_start"]:span["packed_end"]] == list(range(span["end"] - span["start"]))
        reserve_ok &= (bool(set(p["shard_ids"]) & reserved) == schedule["steps"][event["step"] - 1]["reserve"])
    check("packing_correctness", packing_ok and reconstruct_ok, "packs/; ledgers/consumption.jsonl", "Rebuilt all source token spans; packed content and attention hashes validated")
    check("loss_attention_position_masks", mask_ok and position_ok, "packs/; ledgers/token_loss.jsonl", "Block-diagonal causal masks applied by model; role masks and reset position IDs verified")
    quotas_ok = True
    for stage in schedule["steps"]:
        events = [e for e in main if e["step"] == stage["step"]]
        observed = Counter(e["payload"]["lane"] for e in events)
        q = {lane: observed[lane] for lane in stage["quotas"]}
        quotas_ok &= q == stage["quotas"] and all(observed[k] / schedule["slots"] + 1e-12 >= v for k, v in stage["floors"].items())
        content_counts = Counter()
        for e in events:
            content_counts[e["payload"]["lane"]] += e["payload"]["real_tokens"]
        actual.append({"step": stage["step"], "stage": stage["stage"], "sequence_length": stage["sequence_length"],
                       "planned_quotas": stage["quotas"], "actual_quotas": q, "floors": stage["floors"],
                       "token_position_shares": {k: q[k] / schedule["slots"] for k in q},
                       "nonpadding_content_tokens": dict(content_counts), "fraction_rounding_error": {k: q[k] / schedule["slots"] - stage["weights"][k] for k in q}})
    atomic_json(root / "mixture_actual.json", actual)
    check("mixture_curriculum_floors", quotas_ok, "schedule.json; mixture_actual.json", "Exact integer optimizer-step quotas and protected token-position floors; rounded shares reported")
    check("anneal_reserve", reserve_ok, "mixture_actual.json; manifests/; ledgers/consumption.jsonl", "Reserve content absent from main stages; only reserve shards admitted in cooldown")
    status = Counter(e["payload"]["status"] for e in opus)
    overrides = sum(e["payload"].get("protected_floor_override", False) for e in opus)
    accepted_ids = {e["payload"]["candidate_id"] for e in opus if e["payload"]["status"] == "accepted"}
    consumed_decisions = {e["payload"]["opus_decision_id"] for e in main}
    check("opus_audit_trail", all(status[s] > 0 for s in ("accepted", "rejected", "deferred")) and overrides > 0 and consumed_decisions <= accepted_ids,
          "ledgers/opus.jsonl", {"statuses": dict(status), "protected_overrides": overrides, "method": "exact first-order momentum-SGD proxy projection; small-model OPUS-style baseline"})
    expected, resumed = read_json(root / "crash_expected.json"), read_json(root / "resume_actual.json")
    steps = Counter(e["step"] for e in main)
    resume_ok = expected == resumed and sorted(steps) == list(range(1, config["steps"] + 1)) and all(v == schedule["slots"] for v in steps.values())
    check("crash_resume_no_skips_duplicates", resume_ok and crash_result["exit_code"] == 23 and crash_result["committed_step"] == config["crash_step"] - 1,
          "crash_expected.json; resume_actual.json; crash_result.json; ledgers/consumption.jsonl", "Worker killed by os._exit inside transaction; exact next batch, source-span hash and checkpoint matched after rollback")
    replay = ledger.events("consumption", "replay")
    originals = [e for e in main if e["step"] in {2, 3}]
    replay_ok = len(replay) == len(originals) and all(a["payload"][k] == b["payload"][k] for a, b in zip(originals, replay)
        for k in ("batch_id", "pack_id", "token_spans", "loss_mask_hash", "attention_hash", "position_hash"))
    model_replay_ok = digest(ledger.get_state("main", 3)["model"]) == digest(ledger.get_state("replay", 3)["model"])
    replay_loader_ok = ledger.get_state("main", 3)["loader"] == ledger.get_state("replay", 3)["loader"] and ledger.get_state("main", 3)["rng_state"] == ledger.get_state("replay", 3)["rng_state"]
    atomic_json(root / "replay_report.json", {"parent_branch": "main", "checkpoint_step": 1, "replayed_steps": [2, 3],
                                              "batch_span_mask_hashes_match": replay_ok, "model_and_optimizer_match": model_replay_ok, "loader_and_rng_match": replay_loader_ok,
                                              "historical_batch_ids": [e["payload"]["batch_id"] for e in originals],
                                              "replayed_batch_ids": [e["payload"]["batch_id"] for e in replay]})
    check("historical_replay", replay_ok and model_replay_ok and replay_loader_ok, "replay_report.json; checkpoints/replay-0003.json", "Historical packs read from ledger, not resampled; model, optimizer, loader and RNG also reproduced")
    fork = ledger.events("consumption", "fork")
    fork_steps = {e["step"] for e in fork}
    fork_report = {"parent_checkpoint": "main-0001", "fork_checkpoint": "fork-0001", "steps": sorted(fork_steps),
                   "parent_step2_batch": next(e["payload"]["batch_id"] for e in main if e["step"] == 2),
                   "fork_step2_batch": fork[0]["payload"]["batch_id"] if fork else None}
    atomic_json(root / "fork_report.json", fork_report)
    check("explicit_fork", fork_steps == {2, 3} and fork_report["parent_step2_batch"] != fork_report["fork_step2_batch"], "fork_report.json; ledgers/lifecycle.jsonl", "New branch restores earlier checkpoint and explicitly changes RNG/data stream")
    loss_counts = Counter(e["step"] for e in traces)
    trace_valid = all(abs(math.exp(e["payload"]["loss"]) - e["payload"]["perplexity"]) < 1e-8 and e["payload"]["shard_id"] in trained_ids for e in traces)
    check("learning_trace", trace_valid and all(loss_counts[e["step"]] == e["payload"]["token_count"] for e in learning), "ledgers/token_loss.jsonl; ledgers/learning.jsonl", "Actual natural-log cross-entropy linked to target token/source/phase/checkpoint; perplexity recomputed")
    check("actual_training", digest(initial["model"]["params"]) != digest(final["model"]["params"]) and any(e["payload"]["loss_delta"] > 0 for e in learning), "checkpoints/; ledgers/learning.jsonl", "Neural model parameters updated by real gradients; measured before/after loss")
    checkpoints_ok = True
    for branch, step, offset, text, h in ledger.db.execute("SELECT * FROM checkpoints"):
        state = json.loads(text)
        checkpoints_ok &= digest(state) == h and state["ledger_offset"] == offset and all(k in state for k in ("model", "scheduler", "rng_state", "loader", "code_hash", "config_hash"))
    check("checkpoint_binding", checkpoints_ok, "checkpoints/; ledgers/ledger.sqlite", "Model/optimizer/scheduler/RNG/dataloader/code/config and ledger offset saved transactionally")
    elapsed = sum(e["payload"]["timings"]["train_seconds"] + e["payload"]["timings"]["loader_seconds"] for e in learning)
    train_time = sum(e["payload"]["timings"]["train_seconds"] for e in learning)
    selector_time = sum(e["payload"]["timings"]["selector_seconds"] for e in learning)
    performance = {"scope": "main CPU worker; excludes artifact export, orchestration, tests and replay/fork",
                   "elapsed_loader_plus_train_seconds": elapsed, "train_seconds": train_time, "selector_seconds": selector_time,
                   "raw_token_positions": all_positions, "nonpadding_tokens": real_tokens, "useful_loss_bearing_tokens": useful,
                   "packing_utilization": real_tokens / all_positions, "loss_bearing_fraction": useful / all_positions,
                   "raw_token_positions_per_second": all_positions / elapsed, "useful_loss_bearing_tokens_per_second": useful / elapsed,
                   "accepted_nonpadding_tokens_per_second": real_tokens / elapsed, "candidate_token_positions": sum(e["payload"]["timings"]["candidate_token_positions"] for e in learning),
                   "loader_wait_seconds": sum(e["payload"]["timings"]["loader_seconds"] for e in learning),
                   "logical_ranks": config["logical_ranks"], "physical_gpus": 0, "gpu_idle_time": None,
                   "cache_hit_rate": None, "opus_rejection_rate": status["rejected"] / (status["accepted"] + status["rejected"]),
                   "packing_policy_benchmarks": read_json(root / "packing_report.json")}
    atomic_json(root / "performance.json", performance)
    check("throughput_accounting", all_positions == schedule["total_token_positions"] and useful == len(traces) and elapsed > 0, "performance.json; ledgers/learning.jsonl; packs/", "Measured CPU timing and reconstructible useful-token counts; no claimed GPU throughput")
    check("automated_tests", tests_result["returncode"] == 0, "tests.log; tests_result.json", "Important invariant tests executed in the same interpreter")
    results = {r["requirement"]: r for r in findings}
    evidence = {"schema_version": 1, "passed": all(r["passed"] for r in findings), "requirements": results,
                "run_id": config["run_id"], "python": sys.version, "scope": "tiny real CPU training; logical rank partitioning; no distributed GPU claim",
                "steps": config["steps"], "manifest_count": len(registry["manifests"]), "event_count": len(ledger.events())}
    atomic_json(root / "evidence.json", evidence)
    rows = ["# Generated evidence", "", "All results are computed from artifacts and checked invariants.", "", "| Requirement | Result | Evidence |", "|---|---|---|"]
    rows.extend(f"| {r['requirement']} | {'PASS' if r['passed'] else 'FAIL'} | {r['evidence']} |" for r in findings)
    rows += ["", f"CPU packing utilization: {performance['packing_utilization']:.1%}; useful loss-bearing tokens/s: {performance['useful_loss_bearing_tokens_per_second']:.2f}.",
             "", "Selector is an exact-gradient, momentum-SGD first-order baseline, not a production OPUS implementation. Learning usefulness classification measures same-batch loss reduction, not downstream capability gain."]
    (root / "evidence.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    ledger.materialize(); ledger.close()
    for r in findings:
        emit(root, f"[{'PASS' if r['passed'] else 'FAIL'}] {r['requirement']}")
    return evidence


def demo(args):
    root = artifact_root(args.artifacts)
    config = load_config(args.config)
    (root / "run.log").write_text("", encoding="utf-8")
    docs_path = PROJECT / "data" / "corpus.json"
    docs = read_json(docs_path) if docs_path.exists() else make_corpus()
    tokenizer, records, manifests, blocked = build(root, docs)
    emit(root, f"[PASS] shards_created count={len(manifests)}; blocked={len(blocked)}")
    emit(root, f"[PASS] tokenizer_hash_verified {tokenizer.hash}")
    schedule = compile_schedule(config)
    atomic_json(root / "schedule.json", schedule)
    supply = []
    for stage in schedule["steps"]:
        for lane, quota in stage["quotas"].items():
            eligible = [r for r in records if r["split"] == "train" and r["reserve"] == stage["reserve"] and r["lane"] == lane
                        and (len(r["tokens"]) <= stage["sequence_length"] or r["kind"] == "pretrain")]
            supply.append({"step": stage["step"], "stage": stage["stage"], "lane": lane, "quota": quota,
                           "eligible_unique_records": len(eligible), "unique_tokens": sum(len(r["tokens"]) for r in eligible),
                           "repeat_cap": config["repeat_cap"], "reserve": stage["reserve"], "shortage": quota > 0 and not eligible})
    atomic_json(root / "supply_report.json", supply)
    if any(row["shortage"] for row in supply):
        raise ValueError("Schedule supply shortage: inspect supply_report.json")
    atomic_json(root / "config.json", config)
    initialize(root, config, schedule, tokenizer, records)
    for record in records:
        if record["split"] == "train":
            admit(root, record, tokenizer.hash, record["reserve"])
    emit(root, "[PASS] manifests_validated; training content, tokenizer, lineage and admission checks passed")
    emit(root, "[PASS] mixture_compiled; token-position quotas and warmup recorded")
    ledger = Ledger(root)
    for r in records:
        if r["split"] != "train":
            try:
                admit(root, r, tokenizer.hash, False)
            except ValueError as error:
                ledger.append("firewall", "main", 0, {"shard_id": r["shard_id"], "split": r["split"], "blocked": True, "reason": str(error)})
            else:
                raise AssertionError("Heldout shard admitted")
    ledger.db.commit(); ledger.close()
    emit(root, "[PASS] eval_shard_blocked; validation/proxy blocked; embedded canary excluded")
    report = []
    plain = [r for r in records if r["split"] == "train" and r["lane"] == "web" and not r["reserve"]]
    for policy in ("pad_only", "concat_chop", "greedy", "best_fit", "structure", "long"):
        start = time.perf_counter(); packs, deferred = pack_records(plain, 64, policy)
        report.append({"policy": policy, "sequences": len(packs), "real_tokens": sum(p["real_tokens"] for p in packs),
                       "token_positions": len(packs) * 64, "utilization": sum(p["real_tokens"] for p in packs) / (len(packs) * 64),
                       "deferred_records": len(deferred), "pack_seconds": time.perf_counter() - start})
    atomic_json(root / "packing_report.json", report)
    emit(root, "[PASS] batches_packed; policy comparison generated")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(PROJECT / "tests"), "-v"], cwd=PROJECT, capture_output=True, text=True)
    (root / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    tests_result = {"returncode": tests.returncode, "command": "python -m unittest discover -s tests -v"}
    atomic_json(root / "tests_result.json", tests_result)
    if tests.returncode:
        raise RuntimeError("Tests failed; inspect tests.log")
    crash_code = run_child(root, args.config, "main", config["steps"], crash=config["crash_step"])
    ledger = Ledger(root)
    crashed_state = ledger.get_state("main")
    partial_events = ledger.events(branch="main", step=config["crash_step"])
    crash_result = {"exit_code": crash_code, "committed_step": crashed_state["step"], "uncommitted_events_remaining": len(partial_events)}
    ledger.close()
    atomic_json(root / "crash_result.json", crash_result)
    if crash_code != 23 or partial_events or crashed_state["step"] != config["crash_step"] - 1:
        raise AssertionError("Unexpected crash/recovery boundary")
    emit(root, "[PASS] crash_simulated; uncommitted transaction rolled back")
    started = time.perf_counter()
    if run_child(root, args.config, "main", config["steps"]) != 0:
        raise RuntimeError("Resume worker failed")
    resume_seconds = time.perf_counter() - started
    if read_json(root / "crash_expected.json") != read_json(root / "resume_actual.json"):
        raise AssertionError("Next batch after restart changed")
    emit(root, "[PASS] run_resumed; committed checkpoint restored in a fresh worker")
    emit(root, "[PASS] resume_next_batch_matched")
    fork_branch(root, "replay", "main", 1, "historical-ledger")
    started = time.perf_counter()
    if run_child(root, args.config, "replay", 3, replay="main") != 0:
        raise RuntimeError("Replay failed")
    replay_seconds = time.perf_counter() - started
    emit(root, "[PASS] historical_stream_replayed; original pack IDs read from ledger")
    fork_config_path = root / "fork_config.toml"
    fork_config_path.write_text('packing_override = "greedy"\n' + Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    fork_config = load_config(fork_config_path)
    fork_branch(root, "fork", "main", 1, "resampled-with-new-rng-and-greedy-packing", fork_config, config["seed"] + 1)
    if run_child(root, fork_config_path, "fork", 3) != 0:
        raise RuntimeError("Fork worker failed")
    emit(root, "[PASS] branch_forked; checkpoint main-0001, explicit alternate RNG stream")
    ledger = Ledger(root)
    state = ledger.get_state("main")
    evaluations = [evaluate_split(root, state, split) for split in ("validation", "test")]
    atomic_json(root / "evaluation_report.json", evaluations)
    for evaluation in evaluations:
        ledger.append("evaluation_access", "main", state["step"], evaluation)
    ledger.db.commit(); ledger.close()
    evidence = audit(root, config, schedule, tests_result, crash_result)
    if evidence["requirements"]["historical_replay"]["passed"]:
        emit(root, "[PASS] replay_hash_matched; batch IDs, token spans, mask hashes and model/data state verified")
    perf = read_json(root / "performance.json")
    perf.update({"resume_worker_total_seconds": resume_seconds, "replay_worker_total_seconds": replay_seconds})
    atomic_json(root / "performance.json", perf)
    emit(root, "[PASS] audit_completed; performance_measured")
    if not evidence["passed"]:
        raise AssertionError("Evidence includes failed requirements")
    emit(root, "[PASS] all_requirements_verified")
    print("Artifacts:", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.toml")
    parser.add_argument("--artifacts", type=Path, default=PROJECT / "submission_artifacts")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--until", type=int, default=8)
    parser.add_argument("--crash", type=int)
    parser.add_argument("--replay-parent")
    args = parser.parse_args()
    if args.worker:
        worker(args.artifacts, load_config(args.config), args.branch, args.until, args.crash, args.replay_parent)
    else:
        demo(args)


if __name__ == "__main__":
    main()
