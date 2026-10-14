import copy
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from . import VERSION
from .corpus import admit
from .ledger import Ledger
from .model import TinyLM
from .packing import make_pack, pack_records, verify_pack
from .schedule import LANES
from .util import atomic_json, digest, emit, immutable_json, read_json, tuples


def select_candidates(candidates, quotas, floors, keep_fraction):
    ranked = sorted(candidates, key=lambda c: (-c["score"], c["candidate_id"]))
    cutoff = max(1, math.ceil(len(ranked) * keep_fraction))
    preliminary = {c["candidate_id"] for c in ranked[:cutoff]}
    selected, decisions = [], []
    selected_ids, rescued = set(), set()
    slots = sum(quotas.values())
    for lane, count in quotas.items():
        pool = [c for c in ranked if c["lane"] == lane]
        if len(pool) < count:
            raise ValueError(f"Unsatisfied lane supply: {lane}")
        protected = math.ceil(floors.get(lane, 0) * slots - 1e-12)
        for index, candidate in enumerate(pool[:count]):
            selected.append(candidate); selected_ids.add(candidate["candidate_id"])
            if index < protected and candidate["candidate_id"] not in preliminary:
                rescued.add(candidate["candidate_id"])
    for candidate in candidates:
        cid = candidate["candidate_id"]
        accepted = cid in selected_ids
        override = cid in rescued
        decisions.append({k: v for k, v in candidate.items() if k != "pack"} | {
            "status": "accepted" if accepted else "rejected", "protected_floor_override": override,
            "preliminary_global_accept": cid in preliminary,
            "reason": "protected_floor_override" if override else ("lane_quota" if accepted else "quota_pressure"),
            "effective_token_estimate": None})
    # Final deterministic order is independent of worker/rank dispatch order.
    selected.sort(key=lambda c: (LANES.index(c["lane"]), c["candidate_id"]))
    return selected, decisions


def prepare_batch(root, config, schedule, state, model, branch="main"):
    timer = time.perf_counter()
    step = state["step"] + 1
    stage = schedule["steps"][step - 1]
    registry = read_json(Path(root) / "registry.json")
    rng = random.Random(); rng.setstate(tuples(state["rng_state"]))
    length = stage["sequence_length"]
    proxy_records = [r for r in registry["records"] if r["split"] == "proxy"]
    proxy_packs = [make_pack([(r, 0, min(len(r["tokens"]), length))], length, "pad_only") for r in proxy_records]
    proxy_loss, proxy_gradient, _ = model.loss_gradient(proxy_packs)
    lr = config["learning_rate"] / (1 + 0.1 * (step - 1))
    candidates, decisions = [], []
    scoring_seconds = 0.0
    for lane in LANES:
        records = [r for r in registry["records"] if r["lane"] == lane and r["split"] == "train" and r["reserve"] == stage["reserve"]]
        rng.shuffle(records)
        count = stage["quotas"][lane]
        for r in records:
            admit(root, r, registry["tokenizer_hash"], stage["reserve"])
        for r in records:
            if count == 0 or (len(r["tokens"]) > length and r["kind"] != "pretrain"):
                decisions.append({"candidate_id": digest([step, r["doc_id"], "deferred"]), "lane": lane, "shard_ids": [r["shard_id"]],
                                  "status": "deferred", "reason": "stage_inactive" if count == 0 else "structured_overflow",
                                  "stage": stage["stage"], "model_checkpoint": state["checkpoint_id"], "proxy_version": state["proxy_hash"],
                                  "score": None, "protected_floor_override": False, "effective_token_estimate": None})
        if not count:
            continue
        eligible = [r for r in records if len(r["tokens"]) <= length or r["kind"] == "pretrain"]
        if not eligible:
            raise ValueError("No eligible data for " + lane)
        policy = "structure" if lane == "agentic" else ("long" if lane == "long" else config.get("packing_override", "best_fit"))
        # Oversize plain text is explicitly span-chopped; structured examples are never cropped.
        if any(len(r["tokens"]) > length for r in eligible):
            policy = "concat_chop"
        repeated = eligible * (count + 2)
        rng.shuffle(repeated)
        packs, _ = pack_records(repeated, length, policy)
        if len(packs) < count + 1:
            raise ValueError("Insufficient packed candidate supply")
        for index, pack in enumerate(packs[:count + 1]):
            verify_pack(pack)
            before = time.perf_counter()
            loss, gradient, _ = model.loss_gradient([pack])
            score = model.projected_utility(gradient, proxy_gradient, lr, config["momentum"])
            scoring_seconds += time.perf_counter() - before
            candidate = {"candidate_id": digest([step, lane, index, pack["pack_id"]]), "pack": pack, "pack_id": pack["pack_id"],
                         "lane": lane, "shard_ids": sorted({s["shard_id"] for s in pack["spans"]}), "score": score,
                         "candidate_loss": loss, "stage": stage["stage"], "model_checkpoint": state["checkpoint_id"],
                         "proxy_version": state["proxy_hash"]}
            candidates.append(candidate)
    selected, selection = select_candidates(candidates, stage["quotas"], stage["floors"], config["selector_keep_fraction"])
    decisions.extend(selection)
    batch = {"step": step, "stage": stage, "packs": [c["pack"] for c in selected],
             "decision_ids": [c["candidate_id"] for c in selected], "rng_state_after": rng.getstate(),
             "proxy_loss_before": proxy_loss, "learning_rate": lr}
    batch["batch_id"] = digest({"step": step, "stage": stage, "packs": [p["pack_id"] for p in batch["packs"]]})
    return batch, decisions, proxy_packs, {"loader_seconds": time.perf_counter() - timer, "selector_seconds": scoring_seconds,
                                         "candidate_token_positions": len(candidates) * length}


def batch_from_history(root, config, schedule, state, parent, step):
    ledger = Ledger(root)
    events = ledger.events("consumption", parent, step)
    historical_state = ledger.get_state(parent, step)
    ledger.close()
    if len(events) != schedule["slots"]:
        raise ValueError("Historical batch incomplete")
    payloads = [e["payload"] for e in events]
    packs = [read_json(Path(root) / "packs" / (p["pack_id"] + ".json")) for p in payloads]
    for pack in packs:
        verify_pack(pack)
        for span in pack["spans"]:
            record = next(r for r in read_json(Path(root) / "registry.json")["records"] if r["doc_id"] == span["doc_id"])
            admit(root, record, state["tokenizer_hash"], schedule["steps"][step - 1]["reserve"])
            if pack["tokens"][span["packed_start"]:span["packed_end"]] != record["tokens"][span["start"]:span["end"]]:
                raise ValueError("Historical token span mismatch")
    stage = schedule["steps"][step - 1]
    batch = {"step": step, "stage": stage, "packs": packs, "decision_ids": [p["opus_decision_id"] for p in payloads],
             "rng_state_after": historical_state["rng_state"], "proxy_loss_before": None,
             "learning_rate": config["learning_rate"] / (1 + 0.1 * (step - 1))}
    batch["batch_id"] = digest({"step": step, "stage": stage, "packs": [p["pack_id"] for p in packs]})
    if any(p["batch_id"] != batch["batch_id"] for p in payloads):
        raise ValueError("Historical batch hash mismatch")
    return batch


def worker(root, config, branch, until, crash_step=None, replay_parent=None, crash_after_commit=False):
    root = Path(root)
    schedule = read_json(root / "schedule.json")
    ledger = Ledger(root)
    state = ledger.get_state(branch)
    code_hash = digest({p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")})
    if state["config_hash"] != digest(config) or state["code_hash"] != code_hash or state["schedule_hash"] != schedule["hash"]:
        raise ValueError("Checkpoint code/config/schedule identity mismatch")
    if state["corpus_version_hash"] != digest(read_json(root / "registry.json")):
        raise ValueError("Corpus registry changed after checkpoint")
    if state["loader"]["next_step"] != state["step"] + 1:
        raise ValueError("Checkpoint/dataloader step mismatch")
    model = TinyLM.restore(state["model"])
    while state["step"] < until:
        if replay_parent:
            start = time.perf_counter()
            batch = batch_from_history(root, config, schedule, state, replay_parent, state["step"] + 1)
            decisions, proxy_packs = [], []
            timings = {"loader_seconds": time.perf_counter() - start, "selector_seconds": 0.0, "candidate_token_positions": 0}
        else:
            batch, decisions, proxy_packs, timings = prepare_batch(root, config, schedule, state, model, branch)
        step = batch["step"]
        if step == config["crash_step"] and branch == "main":
            expectation = {"step": step, "batch_id": batch["batch_id"], "pack_ids": [p["pack_id"] for p in batch["packs"]],
                           "span_hash": digest([p["spans"] for p in batch["packs"]]), "checkpoint_before": state["checkpoint_id"]}
            if crash_step:
                atomic_json(root / "crash_expected.json", expectation)
            else:
                atomic_json(root / "resume_actual.json", expectation)
        start = time.perf_counter()
        before, gradient, traces = model.loss_gradient(batch["packs"])
        norm = model.update(gradient, batch["learning_rate"], config["momentum"])
        after, _, _ = model.loss_gradient(batch["packs"], gradient=False)
        proxy_after = model.loss_gradient(proxy_packs, gradient=False)[0] if proxy_packs else None
        train_seconds = time.perf_counter() - start
        exposures = Counter(state["loader"]["document_exposures"])
        for pack in batch["packs"]:
            for span in pack["spans"]:
                exposures[span["doc_id"]] += 1
        if max(exposures.values()) > config["repeat_cap"]:
            raise ValueError("Repeated-pass cap exceeded")
        checkpoint_id = f"{branch}-{step:04d}"
        for pack in batch["packs"]:
            pack_path = root / "packs" / (pack["pack_id"] + ".json")
            if pack_path.exists():
                if read_json(pack_path) != pack:
                    raise ValueError("Immutable packed object changed")
            else:
                atomic_json(pack_path, pack)
        ledger.db.execute("BEGIN IMMEDIATE")
        for decision in decisions:
            ledger.append("opus", branch, step, decision)
        for index, pack in enumerate(batch["packs"]):
            rank = index % config["logical_ranks"]
            accumulation = index // config["logical_ranks"]
            ledger.append("consumption", branch, step, {
                "run_id": config["run_id"], "branch_id": branch, "global_step": step,
                "checkpoint_before": state["checkpoint_id"], "checkpoint_after": checkpoint_id,
                "rank": rank, "microbatch_id": accumulation, "sample_index": index, "batch_id": batch["batch_id"],
                "pack_id": pack["pack_id"], "packed_sample_ids": [pack["pack_id"]], "token_spans": pack["spans"],
                "shard_ids": sorted({s["shard_id"] for s in pack["spans"]}), "loss_mask_hash": pack["loss_mask_hash"],
                "attention_hash": pack["attention_hash"], "position_hash": pack["position_hash"],
                "attention_policy": pack["attention_policy"], "position_policy": pack["position_policy"],
                "lane": pack["lane"], "curriculum_stage": batch["stage"]["stage"], "tokenizer_hash": state["tokenizer_hash"],
                "dataloader_version": VERSION, "opus_decision_id": batch["decision_ids"][index],
                "token_positions": len(pack["tokens"]), "real_tokens": pack["real_tokens"],
                "supervised_tokens": sum(pack["loss_mask"]), "repeated_passes": {s["doc_id"]: exposures[s["doc_id"]] for s in pack["spans"]}})
        for trace in traces:
            ledger.append("token_loss", branch, step, {**trace, "model_age_tokens": state["loader"]["token_positions"],
                                                      "stage": batch["stage"]["stage"], "checkpoint_before": state["checkpoint_id"],
                                                      "checkpoint_after": checkpoint_id, "repeat_pass": exposures[trace["doc_id"]],
                                                      "opus_decision_id": batch["decision_ids"][trace["sample_index"]]})
        ledger.append("learning", branch, step, {
            "batch_id": batch["batch_id"], "stage": batch["stage"]["stage"], "loss_before": before, "loss_after": after,
            "loss_delta": before - after, "gradient_norm": norm, "proxy_loss_before": batch["proxy_loss_before"],
            "proxy_loss_after": proxy_after, "usefulness_basis": "same-batch-loss-change; not heldout causal utility",
            "classification": "useful_on_training_batch" if before > after else "nonimproving_on_training_batch",
            "token_count": len(traces), "timings": {**timings, "train_seconds": train_seconds}})
        if step == crash_step:
            emit(root, f"[CRASH] process exits at step {step} after training, before transaction commit")
            os._exit(23)  # real process death; SQLite rolls back its uncommitted writes
        new_state = {
            **state, "model": model.state(), "checkpoint_id": checkpoint_id, "rng_state": batch["rng_state_after"],
            "loader": {"next_step": step + 1, "token_positions": batch["stage"]["token_end"], "document_exposures": dict(exposures)},
            "scheduler": {"step": step, "learning_rate": batch["learning_rate"]},
        }
        state = ledger.save_state(branch, step, new_state)
        ledger.db.commit()
        if crash_after_commit:
            os._exit(24)
        ledger.materialize()
        if decisions:
            emit(root, f"[PASS] opus_decisions_recorded step={step}; candidates={len(decisions)}")
        emit(root, f"[PASS] checkpoint_saved {checkpoint_id}; batch={batch['batch_id'][:12]}; loss={before:.6f}->{after:.6f}")
    ledger.close()


def initialize(root, config, schedule, tokenizer, records):
    ledger = Ledger(root)
    code_hash = digest({p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")})
    model = TinyLM(len(tokenizer.vocabulary), config["hidden_size"], max(s["sequence_length"] for s in schedule["steps"]), config["seed"])
    state = {"model": model.state(), "checkpoint_id": "main-0000", "tokenizer_hash": tokenizer.hash,
             "config_hash": digest(config), "code_hash": code_hash, "schedule_hash": schedule["hash"],
             "corpus_version_hash": digest(read_json(Path(root) / "registry.json")),
             "proxy_hash": digest([r for r in records if r["split"] == "proxy"]), "rng_state": random.Random(config["seed"]).getstate(),
             "loader": {"next_step": 1, "token_positions": 0, "document_exposures": {}}, "scheduler": {"step": 0, "learning_rate": config["learning_rate"]}}
    ledger.db.execute("INSERT INTO branches VALUES('main',NULL,NULL,'scheduled')")
    ledger.append("lifecycle", "main", 0, {"event": "run_initialized", "config_hash": digest(config), "code_hash": code_hash})
    ledger.save_state("main", 0, state); ledger.db.commit(); ledger.materialize(); ledger.close()


def fork_branch(root, name, parent, step, mode, config=None, rng_seed=None):
    ledger = Ledger(root)
    state = ledger.get_state(parent, step)
    if config is not None:
        state["config_hash"] = digest(config)
    if rng_seed is not None:
        state["rng_state"] = random.Random(rng_seed).getstate()
    ledger.db.execute("INSERT INTO branches VALUES(?,?,?,?)", (name, parent, step, mode))
    ledger.append("lifecycle", name, step, {"event": "branch_forked", "parent": parent, "parent_step": step,
                                          "parent_checkpoint_hash": digest(ledger.get_state(parent, step)), "stream_mode": mode,
                                          "config_hash": state["config_hash"], "rng_seed_override": rng_seed})
    state["checkpoint_id"] = f"{name}-{step:04d}"
    ledger.save_state(name, step, state); ledger.db.commit(); ledger.materialize(); ledger.close()
