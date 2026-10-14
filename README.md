# V5 Training Data Execution System

It builds admitted tokenized shards, compiles a curriculum, packs batches, trains a tiny CPU language model, records consumption and learning, kills a worker process, resumes exactly, replays historical batches, forks an earlier checkpoint, and generates auditable evidence. No external dataset, model download, GPU, API key, or third-party Python package is needed.

## Run locally on Windows, macOS, or Linux

Install Python **3.11 or newer**, then open a terminal in the repository root (the folder containing `run_demo.py` and `pyproject.toml`). The demo uses only Python's standard library and runs on CPU. No package installation or environment activation is required.

Run the complete demonstration with one command:

| OS | Command |
|---|---|
| Windows (PowerShell or Command Prompt) | `py -3 run_demo.py` |
| macOS / Linux | `python3 run_demo.py` |

On Windows, if the `py` launcher is unavailable, use `python run_demo.py`. Ensure the selected interpreter is Python 3.11+. The command builds artifacts, runs all invariant tests, trains the model, deliberately crashes a worker, resumes, replays, forks, and audits the results. Outputs are written to `submission_artifacts/`.

### Optional isolated environment

Virtual environments contain machine-specific interpreter paths. Create a new environment on each machine instead of copying or committing `.venv/`.

Windows (PowerShell or Command Prompt):

```text
py -3 -m venv --without-pip .venv
.venv/Scripts/python.exe run_demo.py
```

macOS / Linux:

```sh
python3 -m venv --without-pip .venv
.venv/bin/python run_demo.py
```

If you already have `uv`, this alternative runs from the repository root on all three operating systems:

```sh
uv run python run_demo.py
```

`pyproject.toml` declares project metadata, the Python 3.11+ requirement, and an empty dependency list. The project runs directly from the checkout so the bundled corpus and configuration remain available. `requirements.txt` also documents the zero-dependency environment; no `pip install` is necessary.

### Tests and artifact verification

The complete demo runs the tests automatically. To run only the tests or independently verify existing artifacts:

| Task | Windows | macOS / Linux |
|---|---|---|
| Invariant tests | `py -3 -m unittest discover -s tests -v` | `python3 -m unittest discover -s tests -v` |
| Verify artifacts | `py -3 verify_artifacts.py` | `python3 verify_artifacts.py` |

When using a virtual environment, replace `py -3` or `python3` with its interpreter path shown above. With `uv`, use `uv run python -m unittest discover -s tests -v` or `uv run python verify_artifacts.py`.

The artifact verifier reconstructs source spans, validates masks and the journal chain, checks checkpoint files against committed states, and recomputes resume/replay/throughput invariants without trusting `evidence.json`.

The Windows convenience script is also available: `powershell -NoProfile -File ./setup.ps1`. If Python is not on PATH, pass `-Python C:/path/to/python.exe` to `setup.ps1`.

## Design

```text
authored documents → NFC/Indic-safe cleaning and admission
 → frozen character tokenizer → content-addressed JSON token shards/manifests
 → token-position schedule + lane quotas + anneal reserve
 → policy-specific fixed-length packs + masks + source spans
 → model-gradient proxy selection → logical-rank microbatches
 → real CPU gradient update → consumption/token-loss/learning journal
 → atomic model+optimizer+RNG+loader checkpoint → crash/resume/replay/fork/audit
```

The frozen character tokenizer is intentionally small and transparent, not a production Indic tokenizer. Its vocabulary is fit deterministically on train/selector-development text, never on validation/test text. Seven canonical special tokens are single IDs. Unicode NFC keeps legitimate Indic joiners; raw text containing literal role markers is rejected. The bundled authored corpus covers English, Hindi, Tamil, Telugu, Bengali, code, STEM, reasoning, coherent multi-file text and multi-turn tool-failure/recovery traces. It contains explicit heldout and malicious-overlap fixtures to exercise the firewall.

Every tokenized shard is named by its canonical content SHA-256. Manifests bind source, document IDs, tokenizer, cleaning code, language/script, provenance, license, clean/admission status, split, reserve and fingerprints. Admission is rechecked before sampling. Exact duplicates, unknown licenses, embedded evaluation fingerprints, and all validation/test/proxy shards are blocked from gradient-bearing training. These are meaningful fixture checks, not a claim of industrial PII/near-dedup coverage; imported corpora require stronger upstream cleaning.

Packing implements pad-only, concatenate-and-chop, first-fit greedy, best-fit, structure-preserving and long-context policies. Structured and long examples that exceed a window are deferred, never silently cropped. Independent documents/trajectories get block-diagonal causal attention and reset position IDs. Document first tokens, user/tool/system context and padding have no target loss; assistant text and assistant EOS are supervised. The model actually consumes these masks and position IDs. EOS alone is **not** assumed to isolate attention.

The demo uses eight samples per optimizer step (`2 logical ranks × 1 sample × 4 accumulation rounds`), on a **single CPU process**. Gradients over the global batch are normalised by actual supervised-target count, equivalent to weighted microbatch accumulation; this is not distributed GPU training. Curriculum lengths rise 64→128 to keep the demonstration cheap. Seven capability lanes and Indic/late-agentic floors are enforced through integer quotas with bounded rounding reported explicitly. Floors apply to allocated token positions, including masked context/padding; actual nonpadding content counts and useful-token counts are separately exposed. With tiny batches, fractional weights cannot always be represented exactly.

The model is a trainable embedding + trainable position embedding + causal uniform-attention average + tanh + vocabulary-softmax language model, with analytic gradients and momentum SGD. It is deliberately not called a full Transformer. Finite-difference tests verify the gradients. Cross-entropy and token perplexity are real measurements. “Useful” in its learning ledger means measured reduction on the same training batch; it does **not** establish downstream ability.

The **OPUS-style selector** computes actual candidate and balanced development-proxy gradients, and scores first-order predicted proxy improvement under the current momentum-SGD update. It logs all candidate scores/checkpoint/proxy identities, preliminary global accept/reject, quota acceptance/rejection, stage/length deferral and protected-floor rescue. This is a transparent small-model baseline, not OPUS's scalable AdamW/Muon production algorithm, and no effective-token multiplier is invented. Proxy data provides a gradient direction but never receives an optimizer update.

The SQLite WAL journal is append-only and hash chained. All consumed microbatches, actual token losses, learning metrics, optimizer state and next loader position commit in **one transaction per optimizer step**. A crash before commit rolls back the entire attempt. A crash after commit but before JSON export retains the committed model/data state; checkpoint files can be rematerialised from SQLite. This prevents a ledger-ahead-of-model race. Every demo step is checkpointed; arbitrary checkpoint-frequency/distributed consensus are outside this local implementation.

## The automatic drill

1. Build/admit shards, compile supply/mixture reports, and compare packing policies.
2. Run invariant tests, including tamper checks, mask isolation and both crash boundaries.
3. Train steps 1–2. During step 3, record its expected next-batch fingerprint, train it, write **uncommitted** ledger records, and terminate the worker with `os._exit(23)`.
4. Start a fresh process from committed step 2. It must generate the same step-3 pack IDs, token-span hash and global batch ID. Audit all completed steps for no skipped/duplicated microbatch slots.
5. Restore step 1 into a replay branch. Read steps 2–3 directly from historical consumption records and immutable pack objects, not from a reseeded sampler. Verify batch/span/mask hashes **and model+optimizer state** match.
6. Restore step 1 into a new fork with a recorded RNG change and greedy-packing config. Record the deliberate stream divergence; parent history stays unchanged.
7. Read validation/test for evaluation only, reconstruct source spans, recompute the journal chain, and generate evidence/performance reports. A failed requirement returns a nonzero exit code.

Rerunning the demo preserves an existing recognised artifact directory by moving it to a sibling `.previous-<timestamp>` folder. It never recursively deletes results. Training identities are reproducible on the same Python/runtime/code/config; elapsed times naturally differ. Checkpoints reject code/config/schedule changes unless an explicit fork records them.

## Submission files and GitHub links

The complete demo generates the required files inside **`submission_artifacts/`**, not in the repository root:

- [Execution log](submission_artifacts/run.log)
- [Machine-readable evidence](submission_artifacts/evidence.json)
- [Human-readable evidence](submission_artifacts/evidence.md)
- [Performance report](submission_artifacts/performance.json)

Commit the complete `submission_artifacts/` directory together with the implementation. Exclude `.venv/`, `.test-temp/`, `__pycache__/`, and archived `submission_artifacts.previous-*/` directories.

The assignment form requires four publicly accessible GitHub links: the repository, `submission_artifacts/run.log`, `submission_artifacts/evidence.json`, and `submission_artifacts/evidence.md`. Open each file on GitHub after pushing and copy its browser URL. Check all four links in an incognito window before submitting.

## Generated deliverables

```text
submission_artifacts/
  run.log                       complete milestone/worker log
  evidence.json                 computed requirement checks and evidence paths
  evidence.md                   human-readable PASS/FAIL summary
  manifests/                    provenance/admission records
  shards/                       immutable tokenized objects
  packs/                        immutable token/mask/position/source-span objects
  ledgers/ledger.sqlite         transactional source of truth
  ledgers/*.jsonl               consumption, OPUS, token loss, learning, lifecycle, firewall
  checkpoints/*.json            model, momentum, scheduler, RNG, loader, ledger offset
  tokenizer.json; registry.json; schedule.json; supply_report.json
  mixture_actual.json; packing_report.json; performance.json
  crash_expected.json; resume_actual.json; crash_result.json
  replay_report.json; fork_report.json; evaluation_report.json
  tests.log; tests_result.json
```

Evidence is generated by checks against these objects, not a hardcoded checklist. CPU throughput is measured with `perf_counter`; performance specifies its timing scope and exposes denominators, packing utilization, useful loss-bearing tokens/sec and selection rejection rate. GPU idle time/cache metrics are `null` rather than simulated. Full-run wall time includes extra checkpoint export and audit work outside the main-worker throughput denominator.
