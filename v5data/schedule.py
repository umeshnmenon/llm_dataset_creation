import math
import tomllib
from pathlib import Path
from .util import digest

LANES = ("web", "code", "stem", "indic", "reasoning", "long", "agentic")


def load_config(path):
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def quotas(weights, floors, slots):
    if any(v < 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Mixture must sum to one")
    if any(not 0 <= v <= weights.get(k, 0) for k, v in floors.items()):
        raise ValueError("Floor exceeds target lane weight")
    q = {k: math.floor(weights.get(k, 0) * slots) for k in LANES}
    minimum = {k: math.ceil(floors.get(k, 0) * slots - 1e-12) for k in LANES}
    q = {k: max(q[k], minimum[k]) for k in LANES}
    if sum(minimum.values()) > slots:
        raise ValueError("Impossible protected floors")
    while sum(q.values()) > slots:
        choices = [k for k in LANES if q[k] > minimum[k]]
        k = max(choices, key=lambda k: (q[k] - weights.get(k, 0) * slots, k)); q[k] -= 1
    while sum(q.values()) < slots:
        k = max(LANES, key=lambda k: (weights.get(k, 0) * slots - q[k], k)); q[k] += 1
    return q


def compile_schedule(config):
    slots = config["logical_ranks"] * config["microbatch_size"] * config["gradient_accumulation"]
    steps, token_start, previous = [], 0, None
    for stage in config["stages"]:
        for index in range(stage["steps"]):
            target = {k: stage["weights"].get(k, 0) for k in LANES}
            alpha = 1.0
            warmup = stage.get("warmup_steps", 0)
            if previous and index < warmup:
                alpha = (index + 1) / warmup
                target = {k: (1 - alpha) * previous.get(k, 0) + alpha * target[k] for k in LANES}
                # A newly activated floor takes precedence over an interpolated quota.
                for lane, floor in stage["floors"].items():
                    if target[lane] < floor:
                        deficit = floor - target[lane]; target[lane] = floor
                        donors = [k for k in LANES if k != lane and target[k] > stage["floors"].get(k, 0)]
                        capacity = sum(target[k] - stage["floors"].get(k, 0) for k in donors)
                        if capacity < deficit:
                            raise ValueError("Warmup cannot satisfy floors")
                        for k in donors:
                            target[k] -= deficit * (target[k] - stage["floors"].get(k, 0)) / capacity
            length = stage["sequence_length"]
            allocation = quotas(target, stage["floors"], slots)
            steps.append({"step": len(steps) + 1, "stage": stage["name"], "sequence_length": length,
                          "token_start": token_start, "token_end": token_start + slots * length,
                          "weights": target, "floors": stage["floors"], "quotas": allocation,
                          "reserve": stage["reserve"], "warmup_alpha": alpha})
            token_start += slots * length
        previous = stage["weights"]
    if len(steps) != config["steps"]:
        raise ValueError("Stage durations do not equal total steps")
    return {"steps": steps, "total_token_positions": token_start, "slots": slots, "hash": digest(steps)}
