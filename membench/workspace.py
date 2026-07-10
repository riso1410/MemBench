from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def setup_workspace(instance: dict[str, Any], instances_dir: Path) -> Path:
    spec = instance["workspace"]
    template = (instances_dir / str(spec["template_dir"])).resolve()
    if not template.is_dir():
        raise ValueError(f"workspace template not found: {template}")
    base = Path(os.environ.get("MEMBENCH_ROOT", Path(__file__).resolve().parent.parent)) / "runs" / "workspaces"
    base.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"membench_{instance['instance_id']}_", dir=base))
    workspace = workdir / "repo"
    shutil.copytree(template, workspace)
    _git(workspace, "init", "-q")
    _git(workspace, "add", "-A")
    _git(
        workspace,
        "-c", "user.email=membench@local",
        "-c", "user.name=membench",
        "commit", "-qm", "base",
    )
    return workspace


def workspace_diff(workspace: Path) -> str:
    _git(workspace, "add", "-N", ".")  # make untracked files visible to diff
    return _git(workspace, "diff", "HEAD").stdout


def restore_protected_paths(workspace: Path, instance: dict[str, Any]) -> None:
    protected = instance.get("workspace", {}).get("protected_paths", ["tests"])
    for path in protected:
        _git(workspace, "checkout", "HEAD", "--", str(path))
        # checkout only reverts TRACKED files; agent-CREATED untracked files under
        # a protected path survive and leak into the model_patch. Delete them too.
        # (direct subprocess, no check -> keep ignore-errors semantics)
        subprocess.run(["git", "clean", "-fd", "--", str(path)],
                       cwd=workspace, capture_output=True, text=True)


def apply_patch(workspace: Path, patch: str) -> bool:
    if not patch.strip():
        return False
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=workspace, input=patch, text=True, capture_output=True,
    )
    return result.returncode == 0


def run_tests(workspace: Path, test_ids: list[str], timeout_sec: int = 600) -> dict[str, Any]:
    if not test_ids:
        return {"passed": True, "returncode": 0, "output": "(no tests)", "duration_sec": 0.0}
    cmd = [sys.executable, "-m", "pytest", "-q", *test_ids]
    started = time.time()
    try:
        result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout_sec)
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr)[-4000:],
            "duration_sec": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "returncode": -1,
            "output": "pytest timed out",
            "duration_sec": round(time.time() - started, 3),
        }


def score_workspace(workspace: Path, instance: dict[str, Any]) -> dict[str, Any]:
    oracle = instance.get("oracle", {})
    f2p_ids = list(oracle.get("fail_to_pass", []))
    p2p_ids = list(oracle.get("pass_to_pass", []))
    f2p = run_tests(workspace, f2p_ids)
    p2p = run_tests(workspace, p2p_ids)
    # No fail_to_pass oracle => no ground truth. run_tests trivially "passes" on an
    # empty id list, so a bool verdict here is a false positive (94/140 disagreed
    # with docker). Emit None so nobody mistakes it for a real verdict.
    resolved = None if not f2p_ids else bool(f2p["passed"] and p2p["passed"])
    # Non-authoritative: local pytest, no Docker/oracle isolation. Named
    # *_unverified so no reader mistakes it for the official Docker verdict.
    return {
        "resolved_local_unverified": resolved,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
    }


def extract_diff(text: str) -> str:
    """Pull the last fenced ```diff/```patch block out of a model response."""
    blocks: list[str] = []
    inside = False
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if inside:
                blocks.append("\n".join(current) + "\n")
                current = []
                inside = False
            elif stripped.lower().startswith(("```diff", "```patch")):
                inside = True
            continue
        if inside:
            current.append(line)
    return blocks[-1] if blocks else ""


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result
