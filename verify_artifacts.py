"""Independent read-only verification of a generated run; no model retraining."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from v5data.corpus import admit
from v5data.ledger import Ledger
from v5data.packing import verify_pack
from v5data.util import digest, read_json


def verify(root):
    root = Path(root)
    ledger = Ledger(root)
    ledger.verify()
    schedule = read_json(root / "schedule.json")
    registry = read_json(root / "registry.json")
    records = {r["doc_id"]: r for r in registry["records"]}
    events = ledger.events("consumption", "main")
    counts = Counter(e["step"] for e in events)
    assert sorted(counts) == list(range(1, len(schedule["steps"]) + 1))
    assert all(v == schedule["slots"] for v in counts.values())
    real, supervised, positions = 0, 0, 0
    for e in events:
        p = e["payload"]; pack = read_json(root / "packs" / (p["pack_id"] + ".json"))
        verify_pack(pack)
        assert pack["loss_mask_hash"] == p["loss_mask_hash"]
        for s in pack["spans"]:
            r = records[s["doc_id"]]
            admit(root, r, registry["tokenizer_hash"], schedule["steps"][e["step"] - 1]["reserve"])
            assert pack["tokens"][s["packed_start"]:s["packed_end"]] == r["tokens"][s["start"]:s["end"]]
        real += pack["real_tokens"]; supervised += sum(pack["loss_mask"]); positions += len(pack["tokens"])
    for stage in schedule["steps"]:
        q = Counter(e["payload"]["lane"] for e in events if e["step"] == stage["step"])
        assert {k: q[k] for k in stage["quotas"]} == stage["quotas"]
    traces = ledger.events("token_loss", "main")
    assert len(traces) == supervised
    assert all(abs(math.exp(e["payload"]["loss"]) - e["payload"]["perplexity"]) < 1e-8 for e in traces)
    assert read_json(root / "crash_expected.json") == read_json(root / "resume_actual.json")
    replay = ledger.events("consumption", "replay")
    original = [e for e in events if e["step"] in {2, 3}]
    assert len(replay) == len(original)
    for a, b in zip(original, replay):
        for key in ("pack_id", "batch_id", "token_spans", "loss_mask_hash", "attention_hash", "position_hash"):
            assert a["payload"][key] == b["payload"][key]
    for key in ("model", "loader", "rng_state"):
        assert ledger.get_state("main", 3)[key] == ledger.get_state("replay", 3)[key]
    for branch, step, offset, text, state_hash in ledger.db.execute("SELECT * FROM checkpoints"):
        obj = read_json(root / "checkpoints" / f"{branch}-{step:04d}.json")
        assert obj["hash"] == state_hash == digest(json.loads(text))
        assert obj["state"]["ledger_offset"] == offset
    perf = read_json(root / "performance.json")
    assert (positions, real, supervised) == (perf["raw_token_positions"], perf["nonpadding_tokens"], perf["useful_loss_bearing_tokens"])
    assert abs(perf["useful_loss_bearing_tokens_per_second"] - supervised / perf["elapsed_loader_plus_train_seconds"]) < 1e-8
    ledger.close()
    print(f"Verified independently: {len(events)} main microbatches, {len(traces)} token-loss rows; resume/replay/checkpoint/mask/throughput invariants match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parent / "submission_artifacts")
    verify(parser.parse_args().root)
