# reasoning-gym-v1

Native Verifiers v1 wrapper around Reasoning Gym's deterministic procedural datasets.

```bash
uv run validate reasoning-gym-v1 \
  --taskset.gym arc_1d \
  --taskset.size 10 \
  --runtime.type subprocess
```

The taskset stores each generated entry on the task and reconstructs the seeded generator
for scoring. Reward comes directly from that generator's `score_answer`; validation checks
that the same verifier assigns the gold answer full credit.

Generator-specific configuration is passed through `--taskset.dataset-config`, for example:

```bash
--taskset.dataset-config '{"min_terms": 2, "max_terms": 6}'
```
