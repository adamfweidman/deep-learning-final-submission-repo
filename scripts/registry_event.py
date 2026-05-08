#!/usr/bin/env python3
"""Append a single event row to runs/registry.jsonl.

This is the only writer of the registry. Per docs/design.md sec 7, the
registry is append-only, script-managed, and owned by the main agent
(/auto-research). Subagents must not call this script directly.

Subcommands:
    submit   record an sbatch submission
    close    record a terminal state for an attempt

Each invocation appends exactly one JSON line. Git metadata
(branch, sha, dirty state, changed files, untracked files, diff stat,
workspace hash) is captured automatically so the agent never has to
pass it.

The workspace hash covers the full effective state at the time of
submission, not just tracked unstaged edits:

  - `git diff HEAD` (both staged and unstaged tracked changes), plus
  - the contents of non-tracked files explicitly referenced by
    --config, --sbatch, and --extra-file.

Untracked launch/config files matter because a fresh attempt often
ships in a new YAML or SBATCH that has not been committed yet, and a
diff over tracked files alone would not represent the full attempt.
The script does not blindly hash every untracked file in the tree;
only paths the agent explicitly names. Git metadata collection is
strict: if the command is not run from a valid git repo with a HEAD
commit, the script exits without appending a row.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REGISTRY = Path("runs/registry.jsonl")
VALID_CLOSE_STATES = ("completed", "failed", "timeout", "cancelled", "invalid")


def _run(cmd: list[str], cwd: Path) -> str:
    try:
        res = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise SystemExit(f"failed to run {' '.join(cmd)!r}: {exc}") from exc
    if res.returncode != 0:
        detail = (res.stderr or res.stdout).strip()
        if detail:
            raise SystemExit(f"{' '.join(cmd)} failed: {detail}")
        raise SystemExit(f"{' '.join(cmd)} failed with exit code {res.returncode}")
    return res.stdout


def _repo_root(start: Path) -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"], start).strip()).resolve()


def _default_attempt_id(job_id: str) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "._-") else "-" for ch in job_id.strip()
    ).strip("-")
    return f"slurm-{cleaned or 'unknown'}"


def _is_tracked(repo_root: Path, rel_path: str) -> bool:
    try:
        res = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SystemExit(f"failed to run git ls-files: {exc}") from exc
    if res.returncode == 0:
        return True
    if res.returncode == 1:
        return False
    detail = (res.stderr or res.stdout).strip()
    raise SystemExit(f"git ls-files --error-unmatch failed: {detail}")


def _is_ignored(repo_root: Path, rel_path: str) -> bool:
    try:
        res = subprocess.run(
            ["git", "check-ignore", "-q", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SystemExit(f"failed to run git check-ignore: {exc}") from exc
    if res.returncode == 0:
        return True
    if res.returncode == 1:
        return False
    detail = (res.stderr or res.stdout).strip()
    raise SystemExit(f"git check-ignore failed: {detail}")


def _resolve_path(invocation_cwd: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = invocation_cwd / path
    return path.resolve()


def _capture_relevant_files(
    repo_root: Path,
    invocation_cwd: Path,
    relevant_paths: list[str],
) -> list[dict]:
    """Capture metadata for explicitly named non-tracked files.

    A path is captured only if it (a) was named via --config/--sbatch/
    --extra-file, (b) exists, and (c) is not tracked by git, or is
    outside the repo. Tracked-but-modified files are already covered by
    `git diff HEAD`, so they do not need separate capture.
    """
    captured: list[dict] = []
    seen: set[str] = set()
    for raw in relevant_paths:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        full = _resolve_path(invocation_cwd, raw)
        try:
            if not full.exists():
                raise SystemExit(f"relevant file {raw!r} does not exist")
            if not full.is_file():
                raise SystemExit(f"relevant path {raw!r} is not a regular file")
        except OSError as exc:
            raise SystemExit(f"cannot inspect relevant file {raw!r}: {exc}") from exc
        try:
            rel_repo = str(full.relative_to(repo_root))
        except ValueError:
            rel_repo = None
            display_path = str(full)
            kind = "external"
        else:
            display_path = rel_repo
            if _is_tracked(repo_root, rel_repo):
                continue
            kind = "ignored" if _is_ignored(repo_root, rel_repo) else "untracked"
        try:
            content = full.read_bytes()
        except OSError as exc:
            raise SystemExit(f"cannot read relevant file {raw!r}: {exc}") from exc
        mode = full.stat().st_mode & 0o777
        captured.append(
            {
                "path": display_path,
                "kind": kind,
                "mode": oct(mode),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return captured


def _git_metadata(
    repo_root: Path,
    invocation_cwd: Path,
    relevant_paths: list[str],
) -> dict:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root).strip()
    sha = _run(["git", "rev-parse", "HEAD"], repo_root).strip()
    changed_files = [
        line
        for line in _run(["git", "diff", "--name-only", "HEAD"], repo_root).splitlines()
        if line
    ]
    untracked_files = [
        line
        for line in _run(
            ["git", "ls-files", "--others", "--exclude-standard"], repo_root
        ).splitlines()
        if line
    ]

    diff = _run(["git", "diff", "HEAD"], repo_root)  # staged + unstaged tracked
    diff_stat = _run(["git", "diff", "HEAD", "--shortstat"], repo_root).strip()

    captured = _capture_relevant_files(repo_root, invocation_cwd, relevant_paths)

    hasher = hashlib.sha256()
    hasher.update(diff.encode("utf-8"))
    for entry in captured:
        # Mix path + mode + content sha into the workspace hash so a
        # rename or chmod surfaces as a different hash even when the
        # file body is identical.
        hasher.update(
            f"\n--- CAPTURED {entry['kind']} {entry['path']} mode={entry['mode']} sha256={entry['sha256']} ---\n".encode(
                "utf-8"
            )
        )
    workspace_hash = (
        hasher.hexdigest()[:16] if (diff or captured) else ""
    )

    # Truncate per-file sha256 in the row to 16 chars to match
    # workspace_hash. Full sha256 is still computable from the file on
    # disk if needed.
    captured_short = [
        {**e, "sha256": e["sha256"][:16]} for e in captured
    ]

    return {
        "repo_root": str(repo_root),
        "branch": branch,
        "sha": sha,
        "dirty": bool(changed_files or untracked_files or captured),
        "changed_files": changed_files,
        "untracked_files": untracked_files,
        "diff_stat": diff_stat,
        "workspace_hash": workspace_hash,
        "captured_files": captured_short,
    }


def _timestamp() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _append(registry: Path, row: dict) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _relevant_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    for attr in ("config", "sbatch"):
        v = getattr(args, attr, None)
        if v:
            paths.append(v)
    extras = getattr(args, "extra_file", None) or []
    paths.extend(extras)
    return paths


def _build_row(
    args: argparse.Namespace,
    repo_root: Path,
    invocation_cwd: Path,
) -> dict:
    row: dict = {
        "event": args.event,
        "timestamp": _timestamp(),
    }

    row["attempt_id"] = args.attempt_id or _default_attempt_id(args.job_id)
    row["job_id"] = args.job_id
    row["note"] = args.note
    if getattr(args, "config", None):
        row["config"] = args.config
    if getattr(args, "sbatch", None):
        row["sbatch"] = args.sbatch
    if getattr(args, "output_dir", None):
        row["output_dir"] = args.output_dir
    if getattr(args, "parent_attempt_id", None):
        row["parent_attempt_id"] = args.parent_attempt_id
    extras = getattr(args, "extra_file", None) or []
    if extras:
        row["extra_files"] = list(extras)

    if args.event == "close":
        if args.state not in VALID_CLOSE_STATES:
            raise SystemExit(
                f"--state must be one of {VALID_CLOSE_STATES}, got {args.state!r}"
            )
        row["state"] = args.state
        if args.metrics_path:
            row["metrics_path"] = args.metrics_path
        if args.evidence_label:
            row["evidence_label"] = args.evidence_label

    row["git"] = _git_metadata(repo_root, invocation_cwd, _relevant_paths(args))
    return row


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="registry_event.py",
        description="Append one event row to runs/registry.jsonl.",
    )
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Path to the registry file (default: runs/registry.jsonl).",
    )
    sub = p.add_subparsers(dest="event", required=True)

    base_attempt = argparse.ArgumentParser(add_help=False)
    base_attempt.add_argument(
        "--attempt-id",
        help=(
            "Stable local attempt id. Defaults to slurm-<job-id> when omitted."
        ),
    )
    base_attempt.add_argument("--job-id", required=True)
    base_attempt.add_argument("--config")
    base_attempt.add_argument("--sbatch")
    base_attempt.add_argument("--output-dir")
    base_attempt.add_argument("--parent-attempt-id")
    base_attempt.add_argument(
        "--note",
        required=True,
        help="Required short factual note for the lifecycle event.",
    )
    base_attempt.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help=(
            "Additional path whose non-tracked content should be folded "
            "into workspace_hash. Repeatable."
        ),
    )

    sub.add_parser(
        "submit",
        parents=[base_attempt],
        help="Record an sbatch submission.",
    )

    close = sub.add_parser(
        "close",
        parents=[base_attempt],
        help="Record a terminal state for an attempt.",
    )
    close.add_argument(
        "--state",
        required=True,
        help=f"One of {VALID_CLOSE_STATES}.",
    )
    close.add_argument("--metrics-path")
    close.add_argument("--evidence-label")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    invocation_cwd = Path(os.getcwd()).resolve()
    repo_root = _repo_root(invocation_cwd)
    registry = Path(args.registry)
    if not registry.is_absolute():
        registry = repo_root / registry

    row = _build_row(args, repo_root, invocation_cwd)
    _append(registry, row)
    json.dump(row, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
