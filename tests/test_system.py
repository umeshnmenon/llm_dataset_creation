import copy
import json
import math
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from v5data.corpus import Tokenizer, admit, build, make_corpus, normalize
from v5data.engine import select_candidates
from v5data.ledger import Ledger
from v5data.model import TinyLM
from v5data.packing import attention_matrix, make_pack, pack_records, verify_pack
from v5data.schedule import compile_schedule, load_config, quotas
from v5data.util import atomic_json, digest, read_json

PROJECT = Path(__file__).resolve().parents[1]
TEST_WORK = PROJECT / ".test-temp"
TEST_WORK.mkdir(exist_ok=True)


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = make_corpus()
        cls.tokenizer = Tokenizer.fit(cls.docs)
        cls.records = [cls.tokenizer.record(d) for d in cls.docs if d["split"] == "train" and d["id"] not in {"embedded-test-leak", "duplicate-web", "unknown-license"}]

    def web_pack(self, length=64):
        r = next(r for r in self.records if r["doc_id"] == "web-0")
        return make_pack([(r, 0, len(r["tokens"]))], length, "pad_only")

    def test_indic_safe_normalization(self):
        self.assertEqual(normalize("क\u200dष &amp; \u200bग"), "क\u200dष & ग")
        self.assertEqual(normalize("a\u200cb"), "a\u200cb")

    def test_tokenizer_frozen_and_canonical(self):
        self.assertEqual(self.tokenizer.hash, Tokenizer.fit(list(reversed(self.docs))).hash)
        self.assertEqual(self.tokenizer.encode("é"), self.tokenizer.encode("e\u0301"))
        with self.assertRaises(ValueError):
            self.tokenizer.encode("fake <assistant> role")

    def test_firewall_and_immutable_admission(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as d:
            t, records, manifests, blocked = build(d, self.docs)
            self.assertTrue(any(x["doc_id"] == "embedded-test-leak" for x in blocked))
            self.assertTrue(any(x["doc_id"] == "duplicate-web" for x in blocked))
            for record in records:
                if record["split"] != "train":
                    with self.assertRaises(ValueError):
                        admit(d, record, t.hash, False)
            r = next(r for r in records if r["split"] == "train" and not r["reserve"])
            admit(d, r, t.hash, False)
            with self.assertRaises(ValueError):
                admit(d, r, "wrong-tokenizer", False)
            changed = copy.deepcopy(r); changed["tokens"][1] += 1
            with self.assertRaises(ValueError):
                admit(d, changed, t.hash, False)
            path = Path(d) / "shards" / (r["shard_id"] + ".json")
            payload = read_json(path); payload["records"][0]["tokens"][1] += 1
            atomic_json(path, payload)
            with self.assertRaises(ValueError):
                admit(d, r, t.hash, False)

    def test_reserve_isolation(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as d:
            t, records, _, _ = build(d, self.docs)
            r = next(r for r in records if r["reserve"])
            with self.assertRaises(ValueError):
                admit(d, r, t.hash, False)
            admit(d, r, t.hash, True)

    def test_structure_preserving_masks_and_position_ids(self):
        rs = [r for r in self.records if r["doc_id"] in {"agentic-0", "agentic-1"}]
        packs, deferred = pack_records(rs, 256, "structure")
        self.assertFalse(deferred)
        p = packs[0]; self.assertEqual(len(p["spans"]), 2)
        verify_pack(p)
        self.assertFalse(any(m and role != "assistant" for m, role in zip(p["loss_mask"], p["roles"])))
        boundary = p["spans"][1]["packed_start"]
        self.assertEqual(p["position_ids"][boundary], 0)
        a = attention_matrix(p)
        self.assertFalse(a[boundary][0]); self.assertFalse(a[0][boundary]); self.assertTrue(a[boundary][boundary])
        model = TinyLM(len(self.tokenizer.vocabulary), hidden=3, max_length=256)
        original_loss = model.loss_gradient([p], gradient=False)[0]
        changed = copy.deepcopy(p)
        # Predictions inside second span must be unaffected by first-span tokens.
        for i in range(boundary):
            changed["tokens"][i] = 1
            changed["loss_mask"][i] = False
        original = copy.deepcopy(p)
        for i in range(boundary):
            original["loss_mask"][i] = False
        self.assertEqual(model.loss_gradient([changed], gradient=False)[0], model.loss_gradient([original], gradient=False)[0])

    def test_structured_overflow_never_cropped(self):
        r = next(r for r in self.records if r["doc_id"] == "agentic-0")
        packs, deferred = pack_records([r], 16, "structure")
        self.assertFalse(packs); self.assertEqual(deferred, [r])
        with self.assertRaises(ValueError):
            pack_records([r], 256, "concat_chop")

    def test_concat_keeps_every_token_and_source_offset(self):
        records = [r for r in self.records if r["doc_id"] in {"web-0", "web-1"}]
        packs, _ = pack_records(records, 16, "concat_chop")
        expected = [t for r in records for t in r["tokens"]]
        actual = [t for p in packs for t, origin in zip(p["tokens"], p["origins"]) if origin]
        self.assertEqual(actual, expected)
        self.assertTrue(all(verify_pack(p) for p in packs))

    def test_pack_tamper_detected(self):
        p = self.web_pack(); p["tokens"][0] += 1
        with self.assertRaises(ValueError):
            verify_pack(p)

    def test_schedule_floors_integer_rounding_warmup(self):
        config = load_config(PROJECT / "config.toml")
        s = compile_schedule(config)
        self.assertEqual(len(s["steps"]), 8)
        for stage in s["steps"]:
            self.assertEqual(sum(stage["quotas"].values()), s["slots"])
            for lane, floor in stage["floors"].items():
                self.assertGreaterEqual(stage["quotas"][lane] / s["slots"], floor)
        self.assertEqual(s["steps"][2]["warmup_alpha"], 0.5)
        with self.assertRaises(ValueError):
            quotas({"web": 1}, {"indic": 0.5}, 8)
        with self.assertRaises(ValueError):
            quotas({"web": 0.4}, {}, 8)

    def test_selector_scoring_order_and_floor_override(self):
        candidates = [{"candidate_id": str(i), "lane": lane, "score": score, "pack": {}, "pack_id": str(i)}
                      for i, (lane, score) in enumerate([("web", 10), ("web", 9), ("indic", -1), ("indic", -2)])]
        selected, decisions = select_candidates(candidates, {"web": 1, "indic": 1}, {"indic": 0.5}, 0.25)
        self.assertEqual({c["candidate_id"] for c in selected}, {"0", "2"})
        self.assertTrue(next(d for d in decisions if d["candidate_id"] == "2")["protected_floor_override"])
        self.assertTrue(any(d["status"] == "rejected" for d in decisions))
        with self.assertRaises(ValueError):
            select_candidates(candidates, {"agentic": 1}, {}, 0.25)

    def test_manual_gradient_matches_finite_difference(self):
        p = self.web_pack()
        m = TinyLM(len(self.tokenizer.vocabulary), hidden=3, max_length=64)
        _, gradient, _ = m.loss_gradient([p])
        for key, row, col in [("bias", 0, p["tokens"][3]), ("embedding", p["tokens"][0], 0), ("position", 2, 1), ("output", 1, p["tokens"][4])]:
            original = m.params[key][row][col]; epsilon = 1e-5
            m.params[key][row][col] = original + epsilon
            plus = m.loss_gradient([p], gradient=False)[0]
            m.params[key][row][col] = original - epsilon
            minus = m.loss_gradient([p], gradient=False)[0]
            m.params[key][row][col] = original
            self.assertAlmostEqual(gradient[key][row][col], (plus - minus) / (2 * epsilon), places=7)

    def test_real_training_reduces_fixed_batch_loss(self):
        p = self.web_pack(); m = TinyLM(len(self.tokenizer.vocabulary), hidden=3, max_length=64)
        before, g, traces = m.loss_gradient([p])
        initial_hash = digest(m.params)
        m.update(g, 0.2, 0.8)
        after = m.loss_gradient([p], gradient=False)[0]
        self.assertLess(after, before); self.assertNotEqual(digest(m.params), initial_hash)
        self.assertTrue(all(abs(math.exp(t["loss"]) - t["perplexity"]) < 1e-10 for t in traces))

    def test_optimizer_checkpoint_continuation_exact(self):
        p = self.web_pack(); m = TinyLM(len(self.tokenizer.vocabulary), hidden=3, max_length=64)
        m.update(m.loss_gradient([p])[1], 0.2, 0.8)
        restored = TinyLM.restore(json.loads(json.dumps(m.state())))
        for instance in (m, restored):
            instance.update(instance.loss_gradient([p])[1], 0.1, 0.8)
        self.assertEqual(m.state(), restored.state())

    def test_ledger_append_only_hash_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as d:
            l = Ledger(d)
            l.append("consumption", "main", 1, {"batch_id": "one"})
            state = l.save_state("main", 1, {"model": {"weights": [1]}})
            l.db.commit()
            self.assertEqual(state["ledger_offset"], 1)
            self.assertTrue(l.verify())
            with self.assertRaises(sqlite3.IntegrityError):
                l.db.execute("DELETE FROM events")
            l.db.rollback()
            self.assertEqual(l.get_state("main", 1), state)
            l.close()

    def test_actual_process_death_rolls_back_pending_step(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as d:
            l = Ledger(d); l.append("consumption", "main", 1, {"batch_id": "committed"}); l.save_state("main", 1, {"value": 1}); l.db.commit(); l.close()
            script = "from v5data.ledger import Ledger; import os,sys; l=Ledger(sys.argv[1]); l.db.execute('BEGIN IMMEDIATE'); l.append('consumption','main',2,{'batch_id':'uncommitted'}); l.save_state('main',2,{'value':2}); os._exit(23)"
            child = subprocess.run([sys.executable, "-c", script, d], cwd=PROJECT)
            self.assertEqual(child.returncode, 23)
            l = Ledger(d)
            self.assertEqual(l.get_state("main")["step"], 1)
            self.assertFalse(l.events(step=2)); self.assertTrue(l.verify()); l.close()

    def test_commit_survives_process_death_before_json_export(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as d:
            script = "from v5data.ledger import Ledger; import os,sys; l=Ledger(sys.argv[1]); l.append('consumption','main',1,{'batch_id':'committed'}); l.save_state('main',1,{'value':42}); l.db.commit(); os._exit(24)"
            child = subprocess.run([sys.executable, "-c", script, d], cwd=PROJECT)
            self.assertEqual(child.returncode, 24)
            l = Ledger(d); self.assertEqual(l.get_state("main")["value"], 42); l.materialize()
            self.assertTrue((Path(d) / "checkpoints" / "main-0001.json").exists()); l.close()


if __name__ == "__main__":
    unittest.main()
