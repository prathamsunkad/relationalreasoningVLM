"""Run epoch6, then epoch4 inside one allocation; never submit scheduler jobs.

Only stdlib is imported here: each torchrun child owns GPU initialization and
teardown. A time-limited training child must leave advancing resumable state.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def contract(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def checked_summary(output, plan, mode):
    output = Path(output)
    failures = sorted(output.glob("failure-rank*.json")) + sorted(output.glob("failures-rank*.json"))
    if (output / "failure.json").exists():
        failures.append(output / "failure.json")
    if failures:
        raise RuntimeError("Model has recorded failure: " + ", ".join(str(p) for p in failures))
    summary = read_json(output / "summary.json")
    if summary is None:
        return None
    if summary.get("plan_sha256") != contract(plan):
        raise RuntimeError("Summary belongs to a different plan: " + str(output))
    if summary.get("mode") != mode:
        raise RuntimeError("Summary mode mismatch: " + str(output))
    if not isinstance(summary.get("update"), int) or summary["update"] < 0:
        raise RuntimeError("Invalid summary update: " + str(output))
    if summary.get("complete") and summary.get("pending_validation", False):
        raise RuntimeError("Completed summary still has pending validation: " + str(output))
    if mode == "pilot" and summary.get("complete") and summary.get("passed") is not True:
        raise RuntimeError("Completed pilot did not pass: " + str(output))
    return summary


def brief(summary):
    if summary is None:
        return None
    keys = ("complete", "passed", "mode", "plan_sha256", "update", "examples",
            "stop_reason", "best_loss", "best_checkpoint", "early_stopped")
    return {key: summary[key] for key in keys if key in summary}


def dispatch(campaign_path, source, seconds, mode, *, runner=None, clock=None):
    if runner is None:
        runner = subprocess.run
    if clock is None:
        clock = time.monotonic
    if mode not in ("pilot", "train") or seconds < 300:
        raise ValueError("Invalid allocation mode or runtime budget")
    campaign_path, source = Path(campaign_path).resolve(), Path(source).resolve()
    campaign = json.loads(campaign_path.read_text())
    entries = campaign.get("models", [])
    if [entry.get("name") for entry in entries] != ["epoch6", "epoch4"]:
        raise ValueError("Campaign must run epoch6 then epoch4 exactly once")
    models = []
    for entry in entries:
        plan_path = Path(entry["plan"])
        if not plan_path.is_absolute():
            plan_path = campaign_path.parent / plan_path
        plan_path = plan_path.resolve()
        plan = json.loads(plan_path.read_text())
        world = plan.get('world_size', 4)
        if type(world) is not int or world not in (2, 4):
            raise ValueError('world_size must be two or four GPU ranks')
        output = Path(plan["pilot_output" if mode == "pilot" else "training_output"])
        if not output.is_absolute():
            raise ValueError("Model output paths must be absolute")
        models.append((entry["name"], plan_path, plan, output))
    if len({item[2].get('world_size', 4) for item in models}) != 1:
        raise ValueError('All models in an allocation must use the same world_size')
    if models[0][3] == models[1][3]:
        raise ValueError("Both models cannot share an output directory")
    deadline = clock() + seconds
    root = campaign_path.parent
    state = dict(mode=mode, campaign_sha256=contract(campaign), complete=False,
                 status="inspecting", models={}, invocations=[])
    lock = (root / ".dispatch.lock").open("a")
    acquired = False
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        for name, plan_path, plan, output in models:
            before = checked_summary(output, plan, mode)
            state["models"][name] = brief(before)
            if before and before.get("complete") is True:
                state.update(status="already_complete", current_model=name)
                atomic_json(root / "campaign-progress.json", state)
                continue
            remaining = int(deadline - clock())
            if remaining < 600:
                state.update(status="allocation_boundary", current_model=name,
                             remaining_seconds=max(0, remaining))
                atomic_json(root / "campaign-progress.json", state)
                atomic_json(root / "campaign-summary.json", state)
                if mode == "pilot":
                    raise RuntimeError("Pilot allocation ended before both models passed")
                return state
            checkpoint = output / ".checkpoints" / "last.json"
            if mode == "pilot" and (checkpoint.exists() or (output / "config.json").exists()):
                raise RuntimeError("Incomplete pilot requires a fresh output directory: " + str(output))
            if mode == "train" and before is not None and not checkpoint.exists():
                raise RuntimeError("Incomplete training summary has no resumable checkpoint: " + str(output))
            previous_progress = read_json(output / "progress.json") or {}
            previous_update = before["update"] if before else previous_progress.get("update", 0)
            if not isinstance(previous_update, int) or previous_update < 0:
                raise RuntimeError("Invalid previous progress")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       f"--nproc_per_node={plan.get('world_size', 4)}", str(source / "tools/train_distributed_interleave.py"),
                       "--plan", str(plan_path), "--output", str(output), "--mode", mode,
                       "--seconds", str(remaining)]
            if mode == "train" and checkpoint.exists():
                command.append("--resume")
            invocation = dict(model=name, command=command, previous_update=previous_update,
                              remaining_seconds=remaining, status="running")
            state["invocations"].append(invocation)
            state.update(status="running", current_model=name)
            atomic_json(root / "campaign-progress.json", state)
            result = runner(command, check=False)
            invocation["returncode"] = result.returncode
            if result.returncode != 0:
                invocation["status"] = "failed"
                raise RuntimeError(f"{name} {mode} subprocess exited {result.returncode}")
            after = checked_summary(output, plan, mode)
            if after is None:
                raise RuntimeError("Successful process produced no summary: " + name)
            state["models"][name] = brief(after)
            invocation["status"] = "complete" if after.get("complete") else "checkpointed"
            state.update(status=invocation["status"], current_model=name)
            atomic_json(root / "campaign-progress.json", state)
            if after.get("complete") is not True:
                if mode == "pilot":
                    raise RuntimeError("Pilot did not complete all correctness checks: " + name)
                if not checkpoint.exists() or after["update"] <= previous_update:
                    raise RuntimeError("Incomplete training did not produce an advancing checkpoint: " + name)
                # Epoch6 retains priority; epoch4 cannot start after an incomplete
                # epoch6, even if the child happened to return early.
                state["status"] = "checkpointed_needs_continuation"
                atomic_json(root / "campaign-progress.json", state)
                atomic_json(root / "campaign-summary.json", state)
                return state
        # Re-read every completion, preventing stale or mode-mismatched gate use.
        for name, _, plan, output in models:
            summary = checked_summary(output, plan, mode)
            if not summary or summary.get("complete") is not True:
                raise RuntimeError("Campaign completion missing model: " + name)
            state["models"][name] = brief(summary)
        state.update(complete=True, status="complete", current_model=None)
        atomic_json(root / "campaign-progress.json", state)
        atomic_json(root / "campaign-summary.json", state)
        return state
    except Exception as error:
        if acquired:
            state.update(status="failed", error=str(error))
            atomic_json(root / "campaign-progress.json", state)
            atomic_json(root / "campaign-summary.json", state)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--mode", choices=("pilot", "train"), required=True)
    args = parser.parse_args()
    dispatch(args.campaign, args.source, args.seconds, args.mode)
