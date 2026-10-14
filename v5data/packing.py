from .util import digest


def attention_matrix(pack):
    seg = pack["segments"]
    return [[seg[i] >= 0 and seg[j] == seg[i] and j <= i for j in range(len(seg))] for i in range(len(seg))]


def make_pack(parts, length, policy):
    tokens, mask, roles, positions, segments, origins, spans = [], [], [], [], [], [], []
    for segment, (record, start, end) in enumerate(parts):
        if end - start > length - len(tokens):
            raise ValueError("Pack overflow")
        offset = len(tokens)
        for j in range(start, end):
            tokens.append(record["tokens"][j])
            mask.append(bool(record["loss_mask"][j]) and j > start)
            roles.append(record["roles"][j]); positions.append(j - start); segments.append(segment)
            origins.append({"doc_id": record["doc_id"], "shard_id": record.get("shard_id", "fixture"), "token_offset": j,
                            "language": record["language"], "script": record["script"], "lane": record["lane"]})
        spans.append({"doc_id": record["doc_id"], "shard_id": record.get("shard_id", "fixture"), "start": start, "end": end,
                      "packed_start": offset, "packed_end": len(tokens), "span_id": digest([record["doc_id"], start, end])})
    if not tokens:
        raise ValueError("Empty pack")
    real = len(tokens)
    pad = length - real
    tokens.extend([0] * pad); mask.extend([False] * pad); roles.extend(["pad"] * pad)
    positions.extend([0] * pad); segments.extend([-1] * pad); origins.extend([None] * pad)
    pack = {"tokens": tokens, "loss_mask": mask, "roles": roles, "position_ids": positions, "segments": segments,
            "origins": origins, "spans": spans, "policy": policy, "attention_policy": "block-diagonal-causal",
            "position_policy": "reset-per-independent-span", "real_tokens": real, "padding_tokens": pad,
            "lane": parts[0][0]["lane"], "kind": parts[0][0]["kind"]}
    pack["loss_mask_hash"] = digest(mask)
    pack["attention_hash"] = digest(attention_matrix(pack))
    pack["position_hash"] = digest(positions)
    pack["pack_id"] = digest(pack)
    return pack


def verify_pack(pack):
    expected = {k: v for k, v in pack.items() if k != "pack_id"}
    if digest(expected) != pack["pack_id"]:
        raise ValueError("Packed-sample hash mismatch")
    length = len(pack["tokens"])
    if not all(len(pack[k]) == length for k in ("loss_mask", "roles", "position_ids", "segments", "origins")):
        raise ValueError("Misaligned packed fields")
    if digest(attention_matrix(pack)) != pack["attention_hash"] or digest(pack["loss_mask"]) != pack["loss_mask_hash"] or digest(pack["position_ids"]) != pack["position_hash"]:
        raise ValueError("Mask/position hash mismatch")
    for i, origin in enumerate(pack["origins"]):
        if origin is None and (pack["loss_mask"][i] or pack["tokens"][i] != 0):
            raise ValueError("Padding has loss")
        if pack["kind"] in {"agentic", "sft"} and pack["roles"][i] != "assistant" and pack["loss_mask"][i]:
            raise ValueError("Context/tool token has loss")
        if i == 0 or pack["segments"][i] != pack["segments"][i - 1]:
            if pack["loss_mask"][i]:
                raise ValueError("Cross-document target leak")
    return True


def pack_records(records, length, policy="best_fit"):
    if policy not in {"pad_only", "concat_chop", "greedy", "best_fit", "structure", "long"}:
        raise ValueError("Unknown packing policy")
    parts, deferred = [], []
    for record in records:
        size = len(record["tokens"])
        if size > length:
            if record["kind"] != "pretrain" or policy != "concat_chop":
                deferred.append(record); continue
            parts.extend([(record, i, min(i + length, size)) for i in range(0, size, length)])
        else:
            parts.append((record, 0, size))
    if policy == "pad_only":
        return [make_pack([part], length, policy) for part in parts], deferred
    if policy == "concat_chop":
        # Slice plain text mechanically, retaining exact source offsets and all tokens.
        if any(p[0]["kind"] != "pretrain" for p in parts):
            raise ValueError("Cannot chop a structured trajectory")
        packs, current, used = [], [], 0
        for record, start, end in parts:
            while start < end:
                stop = min(end, start + length - used)
                current.append((record, start, stop)); used += stop - start; start = stop
                if used == length:
                    packs.append(make_pack(current, length, policy)); current, used = [], 0
        if current:
            packs.append(make_pack(current, length, policy))
        return packs, deferred
    if policy == "long":
        return [make_pack([part], length, policy) for part in parts], deferred
    if policy in {"best_fit", "structure"}:
        parts.sort(key=lambda p: (-(p[2] - p[1]), p[0]["doc_id"]))
    bins, used = [], []
    for part in parts:
        size = part[2] - part[1]
        fits = [i for i, n in enumerate(used) if n + size <= length]
        target = min(fits, key=lambda i: length - used[i] - size) if fits and policy in {"best_fit", "structure"} else (fits[0] if fits else len(bins))
        if target == len(bins):
            bins.append([]); used.append(0)
        bins[target].append(part); used[target] += size
    return [make_pack(parts, length, policy) for parts in bins], deferred
