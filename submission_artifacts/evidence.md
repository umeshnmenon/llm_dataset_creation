# Generated evidence

All results are computed from artifacts and checked invariants.

| Requirement | Result | Evidence |
|---|---|---|
| ledger_integrity | PASS | ledgers/ledger.sqlite |
| tokenizer_integrity | PASS | tokenizer.json; manifests/ |
| immutable_shards | PASS | shards/; manifests/ |
| evaluation_validation_firewall | PASS | ledgers/firewall.jsonl; ledgers/consumption.jsonl |
| contamination_firewall | PASS | registry.json |
| packing_correctness | PASS | packs/; ledgers/consumption.jsonl |
| loss_attention_position_masks | PASS | packs/; ledgers/token_loss.jsonl |
| mixture_curriculum_floors | PASS | schedule.json; mixture_actual.json |
| anneal_reserve | PASS | mixture_actual.json; manifests/; ledgers/consumption.jsonl |
| opus_audit_trail | PASS | ledgers/opus.jsonl |
| crash_resume_no_skips_duplicates | PASS | crash_expected.json; resume_actual.json; crash_result.json; ledgers/consumption.jsonl |
| historical_replay | PASS | replay_report.json; checkpoints/replay-0003.json |
| explicit_fork | PASS | fork_report.json; ledgers/lifecycle.jsonl |
| learning_trace | PASS | ledgers/token_loss.jsonl; ledgers/learning.jsonl |
| actual_training | PASS | checkpoints/; ledgers/learning.jsonl |
| checkpoint_binding | PASS | checkpoints/; ledgers/ledger.sqlite |
| throughput_accounting | PASS | performance.json; ledgers/learning.jsonl; packs/ |
| automated_tests | PASS | tests.log; tests_result.json |

CPU packing utilization: 79.9%; useful loss-bearing tokens/s: 759.18.

Selector is an exact-gradient, momentum-SGD first-order baseline, not a production OPUS implementation. Learning usefulness classification measures same-batch loss reduction, not downstream capability gain.
