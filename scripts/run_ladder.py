"""Capability-ladder campaign driver.

Serves each model in the roster (one at a time on the single RTX 5090) under the
SAME served-model-name the harness config expects, then runs the cross-session
memory benchmark (arms none, oracle, oracle_strong) for N seed replicates into
separate MEMBENCH_RUNS dirs. Fully resumable at the (model, seed) grain via a
.ladder_complete marker; run_cross_session self-resumes at the (arm, task) grain.

  python3 scripts/run_ladder.py                 # full campaign (all models, 3 seeds)
  python3 scripts/run_ladder.py --models qwen2.5-coder-7b --seeds 1
  python3 scripts/run_ladder.py --dry-run

NOTE on "seeds": vLLM is nondeterministic even at fixed decoding config, so a
"seed" is just a repeated run into its own RUNS dir -- there is no RNG to set.

GPU/serving gotchas (do not "simplify" away):
  * Never pkill -f "vllm serve" -- kills the ssh session. Use tmux kill-session.
  * vLLM PATH must include the venv bin, else FlashInfer JIT can't find ninja and
    the engine dies.
  * Main model util 0.78 (leaves ~6GB); embed at 0.08 (~2.6GB) co-resides. util
    0.85 OOMs the pair. Serve every model under served-model-name qwen3-coder-30b
    so configs/claude_code_qwen_pectra.toml is unchanged.
  * 32B at 65536 ctx may OOM the KV cache -> we retry once at 32768.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MB = Path(os.environ.get("MEMBENCH_ROOT", Path(__file__).resolve().parent.parent))
HOME = Path.home()

# Served-model-name every model is exposed under, matching claude_model in
# configs/claude_code_qwen_pectra.toml so the harness config never changes.
SERVED_NAME = "qwen3-coder-30b"
CONFIG = "configs/claude_code_qwen_pectra.toml"
REPOS = "deepset-ai/haystack,beeware/briefcase"
ARMS = "none,oracle,oracle_strong"

VLLM_PATH = f"{HOME}/vllm-serve/.venv/bin:/usr/local/cuda/bin:{os.environ.get('PATH','')}"

# Qwen2.5-Coder-Instruct emits Hermes-style <tool_call> JSON -> parser "hermes"
# (registered in vLLM 0.24.0). Qwen3-Coder uses its own "qwen3_coder" parser.
ROSTER = [
    {"name": "qwen3-coder-30b",  "repo": "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit", "parser": "qwen3_coder", "util": 0.78},
    {"name": "qwen2.5-coder-7b",  "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",  "parser": "hermes", "util": 0.78},
    {"name": "qwen2.5-coder-14b", "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ", "parser": "hermes", "util": 0.78},
    {"name": "qwen2.5-coder-32b", "repo": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", "parser": "hermes", "util": 0.78},
]
ROSTER_BY_NAME = {m["name"]: m for m in ROSTER}

LADDER_ROOT = MB / "runs/ladder"
DRIVER_LOG = LADDER_ROOT / "run_ladder.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DRIVER_LOG.open("a") as fh:
        fh.write(line + "\n")


# --- GPU / tmux helpers ---------------------------------------------------------

def gpu_free_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])


def tmux_alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session],
                          capture_output=True).returncode == 0


def tmux_kill(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


def tmux_launch(session: str, shell_cmd: str) -> None:
    # bash -lc so the venv-prefixed PATH export takes effect for vllm.
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_cmd],
                   check=True)


def tmux_pane_tail(session: str, lines: int = 40) -> str:
    r = subprocess.run(["tmux", "capture-pane", "-t", session, "-p", "-S", f"-{lines}"],
                       capture_output=True, text=True)
    return r.stdout


def wait_gpu_released(min_free_mib: int = 28000, timeout: int = 300) -> None:
    # After killing the previous main model, wait until its VRAM is released.
    # We gate on FREE memory (>=28GB) rather than a raw used<1.5GB check because
    # the embed server (~2.6GB) co-resides and would keep "used" above 1.5GB.
    deadline = time.time() + timeout
    while time.time() < deadline:
        free = gpu_free_mib()
        if free >= min_free_mib:
            return
        log(f"  waiting for GPU to free (free={free}MiB, need>={min_free_mib})")
        time.sleep(10)
    log(f"  WARN: GPU still not free after {timeout}s (free={gpu_free_mib()}MiB); proceeding")


def http_ok(url: str) -> bool:
    return subprocess.run(["curl", "-s", "-m", "5", "-o", "/dev/null", url]).returncode == 0


# --- serving --------------------------------------------------------------------

def ensure_embed() -> None:
    if http_ok("http://127.0.0.1:8001/v1/models"):
        log("embed server already up on :8001")
        return
    log("bringing up embed server on :8001")
    cmd = (
        f'PATH="{VLLM_PATH}" vllm serve Qwen/Qwen3-Embedding-0.6B '
        f'--served-model-name qwen3-embedding --host 127.0.0.1 --port 8001 '
        f'--runner pooling --convert embed --max-model-len 4096 '
        f'--gpu-memory-utilization 0.08'
    )
    tmux_kill("embed")
    tmux_launch("embed", cmd)
    for _ in range(60):  # up to 5 min
        if http_ok("http://127.0.0.1:8001/v1/models"):
            log("embed server up")
            return
        time.sleep(5)
    log("ERROR: embed server did not come up in 5 min")
    sys.exit(3)


def serve_model(model: dict, max_len: int) -> bool:
    """Launch the main model in tmux 'vllm' and wait for /health. Returns success."""
    logfile = LADDER_ROOT / model["name"] / "vllm_serve.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'PATH="{VLLM_PATH}" vllm serve {model["repo"]} '
        f'--served-model-name {SERVED_NAME} --host 127.0.0.1 --port 8000 '
        f'--max-model-len {max_len} --gpu-memory-utilization {model["util"]} '
        f'--enable-auto-tool-choice --tool-call-parser {model["parser"]} '
        f'2>&1 | tee -a {logfile}'
    )
    tmux_launch("vllm", cmd)
    log(f"  serving {model['name']} ({model['repo']}) ctx={max_len} "
        f"util={model['util']} parser={model['parser']}")
    deadline = time.time() + 900  # up to 15 min for load + FlashInfer JIT
    while time.time() < deadline:
        if http_ok("http://127.0.0.1:8000/health"):
            log(f"  {model['name']} healthy on :8000")
            return True
        if not tmux_alive("vllm"):
            tail = tmux_pane_tail("vllm", 40)
            oom = any(k in tail.lower() for k in ("out of memory", "oom", "kv cache",
                                                  "no available memory", "cuda error"))
            log(f"  vllm session died before healthy (oom={oom}). tail:\n{tail[-1500:]}")
            return False
        time.sleep(10)
    log(f"  ERROR: {model['name']} not healthy after 15 min")
    tmux_kill("vllm")
    return False


def bring_up_main(model: dict) -> int:
    """Serve the model, retrying 32B at lower ctx on OOM. Returns the ctx used, or 0 on failure."""
    tmux_kill("vllm")
    wait_gpu_released()
    ensure_embed()
    for max_len in (65536, 32768):
        if serve_model(model, max_len):
            return max_len
        log(f"  {model['name']} failed at ctx={max_len}; killing and "
            f"{'retrying at 32768' if max_len == 65536 else 'giving up'}")
        tmux_kill("vllm")
        wait_gpu_released()
    return 0


# --- campaign env for run_cross_session -----------------------------------------

def campaign_env(runs_rel: str) -> dict:
    uid = os.getuid()
    env = {
        **os.environ,
        "PATH": f"{HOME}/.local/bin:{os.environ.get('PATH','')}",
        "MEMBENCH_ROOT": str(MB),
        "SWB_ROOT": str(HOME / "SWE-bench-Live"),
        "DOCKER_BIN": "podman",
        "DOCKER_HOST": f"unix:///run/user/{uid}/podman/podman.sock",
        "MEMBENCH_CONFIG": CONFIG,
        "MEM0_TELEMETRY": "false",
        "MEMBENCH_RUNS": runs_rel,
    }
    return env


def seed_complete(runs_rel: str) -> bool:
    return (MB / runs_rel / ".ladder_complete").exists()


def run_seed(model_name: str, seed: int) -> bool:
    runs_rel = f"runs/ladder/{model_name}/seed{seed}"
    if seed_complete(runs_rel):
        log(f"  [{model_name} seed{seed}] already complete, skipping")
        return True
    (MB / runs_rel).mkdir(parents=True, exist_ok=True)
    log(f"  [{model_name} seed{seed}] running cross-session -> {runs_rel}")
    cmd = [sys.executable, "scripts/run_cross_session.py",
           "--arms", ARMS, "--repos", REPOS]
    r = subprocess.run(cmd, cwd=MB, env=campaign_env(runs_rel))
    if r.returncode == 0:
        (MB / runs_rel / ".ladder_complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        log(f"  [{model_name} seed{seed}] complete")
        return True
    log(f"  [{model_name} seed{seed}] run_cross_session exited {r.returncode} (not marking complete)")
    return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated model names (default: all roster models)")
    ap.add_argument("--seeds", type=int, default=3, help="number of seed replicates (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    names = args.models.split(",") if args.models else [m["name"] for m in ROSTER]
    for n in names:
        if n not in ROSTER_BY_NAME:
            print(f"unknown model: {n} (roster: {list(ROSTER_BY_NAME)})", file=sys.stderr)
            sys.exit(2)
    models = [ROSTER_BY_NAME[n] for n in names]
    seeds = list(range(1, args.seeds + 1))

    if args.dry_run:
        print(f"models: {[m['name'] for m in models]}")
        print(f"seeds:  {seeds}")
        print(f"arms:   {ARMS}")
        print(f"repos:  {REPOS}")
        for m in models:
            for s in seeds:
                rel = f"runs/ladder/{m['name']}/seed{s}"
                print(f"  {m['name']:20} seed{s}  {rel}  "
                      f"{'[done]' if seed_complete(rel) else '[pending]'}")
        return

    log(f"=== ladder campaign start: models={[m['name'] for m in models]} seeds={seeds} ===")
    for model in models:
        # Skip serving entirely if every seed for this model is already complete.
        if all(seed_complete(f"runs/ladder/{model['name']}/seed{s}") for s in seeds):
            log(f"[{model['name']}] all {len(seeds)} seeds complete, skipping serve")
            continue
        log(f"[{model['name']}] bringing up vLLM")
        ctx = bring_up_main(model)
        if ctx == 0:
            log(f"[{model['name']}] FAILED to serve; skipping to next model")
            continue
        if ctx != 65536:
            log(f"[{model['name']}] NOTE: serving at reduced ctx={ctx}")
        for s in seeds:
            run_seed(model["name"], s)
        log(f"[{model['name']}] done; killing vllm")
        tmux_kill("vllm")
        wait_gpu_released()
    log("=== ladder campaign complete ===")


if __name__ == "__main__":
    main()
