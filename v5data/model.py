"""A real tiny neural causal LM with manually differentiated uniform attention.

No learned Q/K attention or transformer-scale claim. Causal attention averages
only permitted token embeddings, then adds a trainable positional embedding,
tanh, and a trainable vocabulary softmax. All trainable arrays receive gradients.
"""
import math
import random
from .packing import attention_matrix


class TinyLM:
    def __init__(self, vocab_size, hidden=6, max_length=128, seed=73):
        rng = random.Random(seed)
        self.vocab_size, self.hidden, self.max_length = vocab_size, hidden, max_length
        self.params = {
            "embedding": [[rng.uniform(-0.08, 0.08) for _ in range(hidden)] for _ in range(vocab_size)],
            "position": [[rng.uniform(-0.02, 0.02) for _ in range(hidden)] for _ in range(max_length)],
            "output": [[rng.uniform(-0.08, 0.08) for _ in range(vocab_size)] for _ in range(hidden)],
            "bias": [[0.0] * vocab_size],
        }
        self.velocity = self.zeros()

    def zeros(self):
        return {k: [[0.0] * len(row) for row in array] for k, array in self.params.items()}

    def state(self):
        return {"vocab_size": self.vocab_size, "hidden": self.hidden, "max_length": self.max_length,
                "params": self.params, "optimizer_velocity": self.velocity}

    @classmethod
    def restore(cls, state):
        model = cls(state["vocab_size"], state["hidden"], state["max_length"])
        model.params = state["params"]; model.velocity = state["optimizer_velocity"]
        return model

    def loss_gradient(self, packs, gradient=True):
        grad = self.zeros() if gradient else None
        traces, loss_sum = [], 0.0
        E, P, W, b = (self.params[k] for k in ("embedding", "position", "output", "bias"))
        for sample_index, pack in enumerate(packs):
            allowed = attention_matrix(pack)
            tokens, flags, seg = pack["tokens"], pack["loss_mask"], pack["segments"]
            for target_pos in range(1, len(tokens)):
                query = target_pos - 1
                if not flags[target_pos] or seg[target_pos] < 0 or seg[target_pos] != seg[query]:
                    continue
                context = [j for j, yes in enumerate(allowed[query]) if yes]
                if not context:
                    raise ValueError("No legal context for loss-bearing target")
                pos = pack["position_ids"][query]
                hidden = [math.tanh(sum(E[tokens[j]][d] for j in context) / len(context) + P[pos][d]) for d in range(self.hidden)]
                logits = [b[0][v] + sum(hidden[d] * W[d][v] for d in range(self.hidden)) for v in range(self.vocab_size)]
                maximum = max(logits)
                exponentials = [math.exp(x - maximum) for x in logits]
                denominator = sum(exponentials)
                probabilities = [x / denominator for x in exponentials]
                true = tokens[target_pos]
                loss = -math.log(max(probabilities[true], 1e-300)); loss_sum += loss
                traces.append({"pack_id": pack["pack_id"], "sample_index": sample_index, "position": target_pos, "query_position": query,
                               "token_id": true, "role": pack["roles"][target_pos], "loss": loss, "perplexity": math.exp(loss),
                               "loss_mask": True, "eos": true == 2, "special": true < 7, **pack["origins"][target_pos]})
                if gradient:
                    dy = probabilities; dy[true] -= 1.0
                    dz = [(sum(W[d][v] * dy[v] for v in range(self.vocab_size))) * (1 - hidden[d] ** 2) for d in range(self.hidden)]
                    for v in range(self.vocab_size):
                        grad["bias"][0][v] += dy[v]
                        for d in range(self.hidden):
                            grad["output"][d][v] += hidden[d] * dy[v]
                    for d in range(self.hidden):
                        grad["position"][pos][d] += dz[d]
                        contribution = dz[d] / len(context)
                        for j in context:
                            grad["embedding"][tokens[j]][d] += contribution
        count = len(traces)
        if not count:
            raise ValueError("Batch contains no supervised target tokens")
        if gradient:
            for array in grad.values():
                for row in array:
                    for i in range(len(row)):
                        row[i] /= count
        return loss_sum / count, grad, traces

    def projected_utility(self, gradient, proxy_gradient, lr, momentum):
        # First-order predicted proxy loss reduction under actual momentum-SGD update.
        return sum(lr * (momentum * self.velocity[k][r][i] + gradient[k][r][i]) * proxy_gradient[k][r][i]
                   for k in self.params for r, row in enumerate(gradient[k]) for i in range(len(row)))

    def update(self, grad, lr, momentum, clip=1.0):
        norm = math.sqrt(sum(x * x for a in grad.values() for row in a for x in row))
        if not math.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        factor = min(1.0, clip / max(norm, 1e-30))
        for k, array in self.params.items():
            for r, row in enumerate(array):
                for i in range(len(row)):
                    self.velocity[k][r][i] = momentum * self.velocity[k][r][i] + factor * grad[k][r][i]
                    row[i] -= lr * self.velocity[k][r][i]
        return norm
