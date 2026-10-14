# Requirement mapping from Session 6

Read the full lesson (17 sections) and transcript (00:00–02:40). The written assignment is the implementation contract; the transcript adds practical failure modes.

| Session requirement | Implementation |
|---|---|
| Frozen tokenizer, immutable shards and manifests | corpus.py; canonical content hashes; runtime admission |
| EOS, masks and target semantics | packing.py; model.py; finite-difference and context-isolation tests |
| Fixed shapes, useful tokens, structured overflow | packing policies; token positions versus supervised tokens |
| Curriculum, warmup, floors, scarcity, reserve | config.toml; schedule.py; supply_report; quota selection |
| Consumption and learning ledgers | hash-chained SQLite events, JSONL exports, per-token actual loss |
| OPUS acceptance/rejection/defer/protected rescue | current-model gradient projection and candidate decision records |
| Complete checkpoints and ledger offsets | single optimizer-step transaction includes full resumable state |
| Deliberate crash, exact resume | real worker process exit before commit; fresh-process regeneration |
| Replay from recorded stream | historical pack/span/mask identities loaded from ledger |
| Fork and explicit differences | branch parent checkpoint, new RNG and config identity |
| Eval registry/firewall and access logs | train/validation/test/proxy permissions; fingerprint overlap; eval access events |
| Performance and evidence | actual counts/timings; generated invariant results and unit-test transcript |

Key transcript details: 00:35–00:40 distinguishes optimizer updates from checkpoints and names optimizer/scheduler/RNG/loader state; 01:15–01:25 discusses structured packing and tokenizer fertility; 01:45 warns that kill time may differ from checkpoint time; 01:50 highlights rejected candidates and floor overrides; 01:55–02:05 explains natural-log token cross-entropy/perplexity and source-linked phase traces; 02:20–02:25 states the assignment/demo artifacts; 02:35 insists that replay read historical ledger order instead of merely rerunning a seed.

The transcript allows fake training, but this implementation does real small-model gradient updates for stronger evidence. It does not adopt incorrect informal statements literally: attention isolation comes from an explicit mask, padding receives no target loss, and perplexity is exp(mean natural-log cross-entropy), not a general benchmark for intelligence.
