"""Print the exact command that reproduces a relational run, rebuilt from its saved args.json, plus its job-log lines.

    python3 tools/rerun_cmd.py lora_lf_q_int_ft276k
    python3 tools/rerun_cmd.py lora_lf_q_int_ft276k --remap /home/hice1/psunkad3/scratch=$HOME/scratch
    python3 tools/rerun_cmd.py --list                    # every run with a saved args.json

Every argument the run used is spelled out (defaults included), so the command does not depend on later default
changes. Run it from relational/ inside the container (see 00_START_HERE/REPRODUCE.md). Stage 0 slot runs saved no
args.json: their job-log lines carry the flags (launched with relational/slurm/stage0.sbatch).
Paths in args.json are the original PACE paths; --remap old=new rewrites them (repeatable).
"""
import argparse
import json
import os
import re
import shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = f"{ROOT}/results/relational/run_records"
LOG = f"{ROOT}/results/relational/job_log/night_queue_2026-11-08.txt"


def parser_spec(script):
    """Names of the script's arguments, which are flags (store_true) and which repeat (append)."""
    src = open(f"{ROOT}/relational/{script}").read()
    names, flags, appends = set(), set(), set()
    for m in re.finditer(r'add_argument\(\s*"--([A-Za-z0-9_]+)"([^\n]*)', src):
        names.add(m.group(1))
        if 'action="store_true"' in m.group(2):
            flags.add(m.group(1))
        if 'action="append"' in m.group(2):
            appends.add(m.group(1))
    return names, flags, appends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--remap", action="append", default=[], help="old=new path prefix")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.run:
        for r in sorted(os.listdir(RR)):
            if os.path.exists(f"{RR}/{r}/args.json"):
                print(r)
        return
    p = f"{RR}/{a.run}/args.json"
    script = "train_stage1.py"
    if not os.path.exists(p):
        print(f"# {a.run}: no args.json saved (Stage 0 slot runs); see the job-log lines below")
        args = None
    else:
        args = json.load(open(p))
    if args is not None:
        names, flags, appends = parser_spec(script)
        parts = ["python", script]
        skipped = []
        for k, v in args.items():
            if k not in names:
                skipped.append(k)
                continue
            if k in flags:
                if v:
                    parts.append(f"--{k}")
                continue
            if v is None:
                continue
            vals = v if isinstance(v, list) else [v]
            if k in appends:
                for x in vals:
                    parts += [f"--{k}", str(x)]
            else:
                parts += [f"--{k}"] + [str(x) for x in vals]
        cmd = " ".join(shlex.quote(x) for x in parts)
        for m in a.remap:
            old, new = m.split("=", 1)
            cmd = cmd.replace(old, new)
        print("# from", os.path.relpath(p, ROOT))
        if skipped:
            print("# not passed (saved for information only):", ", ".join(skipped))
        print(cmd)
    print("\n# job-log lines for this run (job id, flags, why it was submitted):")
    if os.path.exists(LOG):
        for line in open(LOG):
            f = line.split()
            if len(f) > 1 and f[1] == a.run:
                print(line.rstrip())


if __name__ == "__main__":
    main()
