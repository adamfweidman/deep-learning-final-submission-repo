# scripts/

## `registry_event.py`

The only writer of `runs/registry.jsonl`. Append-only. Captures git
metadata automatically so the agent only has to pass the
event-specific fields.

Subcommands:

- `submit` — record an `sbatch` submission. Required: `--job-id`,
  `--note`. Optional: `--attempt-id` (defaults to `slurm-<job-id>`),
  `--config`, `--sbatch`, `--output-dir`, `--parent-attempt-id`,
  `--extra-file` (repeatable).
- `close` — record a terminal state. Required: `--job-id`, `--note`,
  `--state` (`completed|failed|timeout|cancelled|invalid`).
  Optional: `--attempt-id` (defaults to `slurm-<job-id>`),
  `--metrics-path`, `--evidence-label`, `--output-dir`,
  `--extra-file`.

Example:

```bash
python scripts/registry_event.py submit \
  --job-id 7946325 \
  --config configs/example_run.yaml \
  --sbatch sbatch/run_experiment.SBATCH \
  --output-dir /scratch/$USER/runs/example/2026-05-04-001 \
  --note "First exploratory run."
```

### Workspace hashing

The script captures the **full effective workspace state** for each
event, not just unstaged tracked edits:

- `git diff HEAD` — both staged and unstaged tracked changes.
- Contents of explicitly named non-tracked files passed via
  `--config`, `--sbatch`, and `--extra-file`. This includes
  untracked or ignored files inside the repo and explicitly named
  files outside the repo. A fresh attempt often ships in a new YAML
  or SBATCH that has not been committed yet, and a diff over tracked
  files alone would not represent the run.

The script does **not** blindly hash every untracked file in the tree
(scratch symlinks, logs, `.venv/`, outputs). Only paths the agent
explicitly names are folded in.

Git metadata collection is strict. The script resolves the repo root
with `git rev-parse --show-toplevel` and exits without appending if
the invocation is outside a git repo or the repo has no `HEAD`
commit.

Recorded git fields per event:

- `branch`, `sha`, `dirty`
- `changed_files` — tracked files with staged or unstaged changes
- `untracked_files` — full list of untracked files (gitignore-aware)
- `diff_stat` — `git diff HEAD --shortstat` for human reading
- `workspace_hash` — 16-char sha256 prefix over the diff plus the
  contents of captured non-tracked files (with their paths, kind,
  and modes mixed in, so a rename or chmod surfaces as a different
  hash even when bytes are identical)
- `captured_files` — per-file `{path, kind, mode, size, sha256}` for
  the relevant non-tracked files folded into the hash

### Hard rules baked in

- One event per invocation.
- Append-only. Closed rows are never mutated. Record interpretations,
  corrections, and decision updates in `artifacts/EXPERIMENTS.md`,
  not as separate registry notes.
- No `cell_id`, no `is_recovery`. Use `parent_attempt_id` plus `note`
  for recovery semantics, per `docs/design.md` sec 7.

`/auto-research` invokes this script. Subagents must not.
