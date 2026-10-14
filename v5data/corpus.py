"""Bundled authored fixtures; no claims of production-quality curation."""
import html
import re
import unicodedata
from pathlib import Path
from .util import atomic_json, digest, file_hash, immutable_json, read_json

SPECIALS = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>", "<tool>", "<system>"]
ALLOWED_LICENSES = {"CC0-1.0", "MIT", "CC-BY-4.0"}


def normalize(text):
    text = unicodedata.normalize("NFC", html.unescape(text))
    # Keep U+200C/U+200D, meaningful joiners in Indic scripts.
    text = "".join(c for c in text if c in "\n\t\u200c\u200d" or unicodedata.category(c) not in {"Cc", "Cf"})
    return re.sub(r"[ \t]+", " ", text).strip()


def texts(doc):
    return [m["text"] for m in doc.get("messages", [])] or [doc["text"]]


def joined(doc):
    return "\n".join(normalize(t) for t in texts(doc))


def fingerprints(text, size=18):
    text = re.sub(r"\s+", " ", normalize(text)).casefold()
    return {digest(text[i:i + size]) for i in range(max(1, len(text) - size + 1))}


def make_corpus():
    examples = {
        "web": ["The river flows past the town.", "A healthy tree needs light and water.", "Birds build nests in spring.", "The bus arrives before noon."],
        "code": ["def add(a, b):\n    return a + b", "for n in range(3):\n    print(n)", "assert len([1, 2]) == 2", "items = sorted([3, 1, 2])"],
        "stem": ["Water freezes at zero Celsius.", "Force equals mass times acceleration.", "A cell contains genetic material.", "A triangle has three sides."],
        "indic": ["यह पेड़ हरा है। पानी साफ है।", "காலை வணக்கம். இது ஒரு மரம்.", "ఇది ఒక పుస్తకం. నీరు అవసరం.", "বাংলা একটি ভাষা। বই পড়ি।"],
        "reasoning": ["Three bags of four give twelve: 3 * 4 = 12.", "If x + 2 = 5, then x = 3. Check: 3 + 2 = 5.", "A square has four equal sides. Its area is s * s.", "Two even numbers sum to an even number: 2a + 2b = 2(a+b)."],
        "long": ["File A exports add. File B imports add. " * 2 + "Check both files before an edit.", "Chapter one names a red key. " * 2 + "In chapter two, the red key opens the box.", "Module X writes a value. Module Y reads it. " * 2, "The plan has three stages. Read the opening definition before the final test. "],
    }
    docs = []
    for lane, entries in examples.items():
        for i, text in enumerate(entries):
            docs.append({"id": f"{lane}-{i}", "lane": lane, "text": text, "split": "train", "reserve": False,
                         "kind": "long" if lane == "long" else "pretrain", "language": ["hin", "tam", "tel", "ben"][i] if lane == "indic" else "eng",
                         "script": ["Devanagari", "Tamil", "Telugu", "Bengali"][i] if lane == "indic" else "Latin",
                         "tier": "verified_native" if lane == "indic" else "authored", "license": "CC0-1.0",
                         "source": "bundled-author-created-demo", "difficulty": min(i, 3), "reasoning_band": "short"})
    for i in range(4):
        docs.append({"id": f"agentic-{i}", "lane": "agentic", "kind": "agentic", "split": "train", "reserve": False,
                     "language": "eng", "script": "Latin", "tier": "authored", "license": "CC0-1.0", "source": "bundled-author-created-demo",
                     "difficulty": 3, "reasoning_band": "medium", "messages": [
                         {"role": "user", "text": f"Read file {i}."}, {"role": "assistant", "text": f"read({i})"},
                         {"role": "tool", "text": "Not found."}, {"role": "assistant", "text": f"list(); read({i+1})"},
                         {"role": "tool", "text": "ok"}, {"role": "assistant", "text": "Done."}]})
    # Reserve is distinct content, not a flag pasted onto a main-run duplicate.
    for doc in list(docs):
        if doc["id"].endswith("-0") or doc["id"] == "indic-1":
            reserve = {**doc, "id": "reserve-" + doc["id"], "reserve": True}
            if "messages" in doc:
                reserve["messages"] = [{**m, "text": m["text"].replace("file 0", "file 8").replace("read(0)", "read(8)").replace("read(1)", "read(9)")} for m in doc["messages"]]
            else:
                reserve["text"] = "Review: " + doc["text"] + " Verified."
            if doc["lane"] == "long":
                reserve["text"] = "File C exports read. File D imports read. Change both files together. Verify the caller and the exported function."
            docs.append(reserve)
    for split, text in [("validation", "Heldout validation: violet clocks tick slowly."), ("test", "EVAL_CANARY_73: amber owls guard the gate.")]:
        docs.append({"id": split, "lane": "web", "text": text, "kind": "pretrain", "split": split, "reserve": False,
                     "language": "eng", "script": "Latin", "tier": "authored", "license": "CC0-1.0", "source": "authored-heldout"})
    for i, text in enumerate(["The river crosses a quiet town.", "यह पानी साफ है।", "Read the tool result and retry."]):
        docs.append({"id": f"proxy-{i}", "lane": ["web", "indic", "agentic"][i], "text": text, "kind": "pretrain", "split": "proxy", "reserve": False,
                     "language": "hin" if i == 1 else "eng", "script": "Devanagari" if i == 1 else "Latin", "tier": "authored",
                     "license": "CC0-1.0", "source": "authored-selector-development"})
    leak = {**docs[0], "id": "embedded-test-leak", "text": "Unrelated prefix. EVAL_CANARY_73: amber owls guard the gate. Suffix."}
    docs.extend([leak, {**docs[0], "id": "duplicate-web"}, {**docs[0], "id": "unknown-license", "license": "unknown", "text": "No eligible license was supplied."}])
    return docs


class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.ids = {t: i for i, t in enumerate(vocabulary)}
        self.hash = digest({"version": "nfc-character-v1", "vocabulary": vocabulary})

    @classmethod
    def fit(cls, docs):
        chars = sorted({c for d in docs if d["split"] in {"train", "proxy"} for t in texts(d) for c in normalize(t)})
        return cls(SPECIALS + chars)

    def encode(self, text):
        if any(t in text for t in SPECIALS):
            raise ValueError("Literal conversation special token in content")
        return [self.ids.get(c, 1) for c in normalize(text)]

    def record(self, doc):
        tokens, mask, roles = [], [], []
        messages = doc.get("messages") or [{"role": "document", "text": doc["text"]}]
        for message in messages:
            role = message["role"]
            if role != "document":
                tokens.append(self.ids["<" + role + ">"]); mask.append(False); roles.append(role)
            body = self.encode(message["text"])
            enabled = role in {"document", "assistant"}
            tokens.extend(body); mask.extend([enabled] * len(body)); roles.extend([role] * len(body))
            tokens.append(2); mask.append(enabled); roles.append(role)
        mask[0] = False  # no previous context for the first document token
        return {"doc_id": doc["id"], "tokens": tokens, "loss_mask": mask, "roles": roles,
                "lane": doc["lane"], "kind": doc["kind"], "language": doc["language"], "script": doc["script"],
                "reserve": doc["reserve"], "split": doc["split"], "tier": doc["tier"]}


def build(root, docs):
    root = Path(root)
    tokenizer = Tokenizer.fit(docs)
    atomic_json(root / "tokenizer.json", {"vocabulary": tokenizer.vocabulary, "hash": tokenizer.hash})
    pipeline_hash = file_hash(__file__)
    heldout = [d for d in docs if d["split"] in {"test", "validation", "proxy"}]
    forbidden = set().union(*(fingerprints(joined(d)) for d in heldout))
    blocked, records, manifests, seen = [], [], [], set()
    for doc in docs:
        text_hash = digest(joined(doc))
        reason = None
        if doc["license"] not in ALLOWED_LICENSES:
            reason = "unsafe_or_unknown_license"
        elif doc["split"] == "train" and fingerprints(joined(doc)) & forbidden:
            reason = "heldout_fingerprint_overlap"
        elif doc["split"] == "train" and text_hash in seen:
            reason = "duplicate_clean_document"
        if reason:
            blocked.append({"doc_id": doc["id"], "reason": reason, "content_hash": text_hash}); continue
        if doc["split"] == "train":
            seen.add(text_hash)
        record = tokenizer.record(doc)
        shard_payload = {"tokenizer_hash": tokenizer.hash, "records": [record]}
        shard_name = immutable_json(root / "shards", shard_payload)
        shard_id = shard_name[:-5]
        record["shard_id"] = shard_id
        record["text_fingerprints"] = sorted(fingerprints(joined(doc)))
        records.append(record)
        manifest = {"shard_id": shard_id, "content_hash": shard_id, "source_ids": [doc["source"]], "document_ids": [doc["id"]],
                    "tokenizer_hash": tokenizer.hash, "token_count": len(record["tokens"]), "language": doc["language"], "script": doc["script"],
                    "capability_lane": doc["lane"], "license": doc["license"], "provenance_tier": doc["tier"],
                    "cleaning_pipeline_hash": pipeline_hash, "deduplicated": True, "pii_screened": True,
                    "language_validated": True, "contamination_scanned": True, "eval_overlap": False,
                    "split": doc["split"], "never_train": doc["split"] != "train", "reserve": doc["reserve"],
                    "parent_shard_ids": [], "fingerprints": record["text_fingerprints"],
                    "benchmark_id": doc["id"] if doc["split"] != "train" else None, "version": "fixture-v1"}
        # Fixture admission includes real simple screens, not public-corpus certification.
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", joined(doc)):
            raise ValueError("Fixture contains unredacted email")
        script_ranges = {"hin": (0x0900, 0x097F), "tam": (0x0B80, 0x0BFF), "tel": (0x0C00, 0x0C7F), "ben": (0x0980, 0x09FF)}
        if doc["language"] in script_ranges:
            low, high = script_ranges[doc["language"]]
            if not any(low <= ord(c) <= high for c in joined(doc)):
                raise ValueError("Language/script mismatch")
        atomic_json(root / "manifests" / (shard_id + ".json"), manifest)
        manifests.append(manifest)
    atomic_json(root / "registry.json", {"records": records, "manifests": manifests, "blocked": blocked,
                                          "heldout_fingerprints": sorted(forbidden), "tokenizer_hash": tokenizer.hash,
                                          "cleaning_pipeline_hash": pipeline_hash})
    return tokenizer, records, manifests, blocked


def admit(root, record, tokenizer_hash, allow_reserve):
    manifest = read_json(Path(root) / "manifests" / (record["shard_id"] + ".json"))
    shard = read_json(Path(root) / "shards" / (record["shard_id"] + ".json"))
    if digest(shard) != manifest["content_hash"] or manifest["shard_id"] != digest(shard):
        raise ValueError("Shard hash mismatch")
    if shard["tokenizer_hash"] != tokenizer_hash or manifest["tokenizer_hash"] != tokenizer_hash:
        raise ValueError("Tokenizer hash mismatch")
    if manifest["never_train"] or manifest["split"] != "train":
        raise ValueError("Evaluation/validation/proxy shard blocked")
    if manifest["license"] not in ALLOWED_LICENSES or not all(manifest[k] for k in ("deduplicated", "pii_screened", "language_validated", "contamination_scanned")) or manifest["eval_overlap"]:
        raise ValueError("Admission contract failed")
    if manifest["cleaning_pipeline_hash"] != file_hash(__file__):
        raise ValueError("Cleaning code changed after admission")
    if manifest["reserve"] != allow_reserve:
        raise ValueError("Reserve isolation failed")
    registry = read_json(Path(root) / "registry.json")
    if set(record["text_fingerprints"]) & set(registry["heldout_fingerprints"]):
        raise ValueError("Heldout overlap blocked at sampling time")
    original = shard["records"][0]
    if any(record[k] != original[k] for k in original):
        raise ValueError("In-memory token record differs from immutable shard")
    if manifest["token_count"] != len(record["tokens"]):
        raise ValueError("Manifest token count mismatch")
    return manifest
