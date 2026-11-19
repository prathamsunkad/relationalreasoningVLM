"""Stronger-branch comparison (user, Nov 15: 'Yes, both'): wait until each 50-epoch-branch alignment run levels off,
then launch its Stage 2 branch-dropout run from the alignment's best.pt, with the same rule and recipe used for the
10-epoch-branch runs (lora_w5d_q_*_L6ft).

Levelled = at least 3 validations and the best did not rise by more than 1 point over the last two (or the run
finished). After launching, the alignment run drops to priority 1 and keeps going, as the earlier ones did.
Runs detached on the login node (nohup setsid) so it survives the Claude session; log in logs/auto_stage2_*.log."""
import glob
import json
import os
import re
import shutil
import subprocess
import time

R = "/home/hice1/psunkad3/scratch/relvlm_runs"
A = f"{R}/auto"
PAIRS = [("align_q_int_L6ft50_lr3e-4", "int", "lora_w5d_q_int_L6ft50", "lora_w5d_q_int_L6ft"),
         ("align_q_fuse_L6ft50_lr3e-4", "fuse", "lora_w5d_q_fuse_L6ft50", "lora_w5d_q_fuse_L6ft")]
DEADLINE = time.time() + 4 * 86400
LOG = open(f"{R}/logs/auto_stage2_{time.strftime('%m%d_%H%M%S')}.log", "a", buffering=1)


def log(*a):
    print(time.strftime("%b %d %H:%M:%S"), *a, file=LOG)


def vals(tag):
    out = {}
    for f in glob.glob(f"{R}/logs/stage1_{tag}_*.log"):
        for m in re.finditer(r"^\[val\] step (\d+) pgm: llm_acc ([\d.]+) aux_acc ([\d.]+)", open(f, errors="replace").read(), re.M):
            out[int(m[1])] = (float(m[2]), float(m[3]))
    return [(s, *out[s]) for s in sorted(out)]


def in_queue(tag):
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", "psunkad3"), "-h", "-o", "%j"],
                       capture_output=True, text=True, timeout=60).stdout.split()
    return tag in q


def crashed(tag):
    fs = sorted(glob.glob(f"{R}/logs/stage1_{tag}_*.log"), key=os.path.getmtime)
    return bool(fs) and "Traceback" in open(fs[-1], errors="replace").read() and not in_queue(tag)


def set_priority(tag, n):
    p = f"{R}/priority.json"
    shutil.copy(p, f"{R}/" + time.strftime(".priority_backup_%m%d_%H%M%S.json"))
    d = json.load(open(p)); d[tag] = n
    json.dump(d, open(p, "w"), indent=2)
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", "psunkad3"), "-h", "-t", "PD", "-n", tag, "-o", "%i"],
                       capture_output=True, text=True, timeout=60).stdout.split()
    for j in q:
        subprocess.run(["scontrol", "update", f"JobId={j}", f"Nice={n}"], capture_output=True, timeout=60)


def snapshot(align):
    """Copy best.pt while making sure it was not rewritten during the copy."""
    os.makedirs(f"{A}/snap", exist_ok=True)
    for _ in range(5):
        b1, m1 = open(f"{R}/{align}/best.json").read(), os.path.getmtime(f"{R}/{align}/best.pt")
        step = json.loads(b1)["step"]
        dst = f"{A}/snap/init_{align}_step{step}.pt"
        shutil.copy2(f"{R}/{align}/best.pt", dst)
        if open(f"{R}/{align}/best.json").read() == b1 and os.path.getmtime(f"{R}/{align}/best.pt") == m1:
            return dst, json.loads(b1)
        time.sleep(60)
    raise RuntimeError(f"{align}/best.pt kept changing")


todo = list(PAIRS)
log("start; pairs:", [p[0] for p in todo])
while todo and time.time() < DEADLINE:
    for pair in list(todo):
        align, mode, s2, twin = pair
        if os.path.exists(f"{R}/{s2}"):
            log(f"{s2} already exists; skipping"); todo.remove(pair); continue
        try:
            v = vals(align)
            if crashed(align):
                log(f"{align} CRASHED (Traceback, not in queue) after vals {v}; not launching {s2}")
                todo.remove(pair); continue
            fin = os.path.exists(f"{R}/{align}/final.pt")
            lev = len(v) >= 3 and max(x[1] for x in v[-2:]) - max(x[1] for x in v[:-2]) <= 0.01
            if not (lev or (fin and v)):
                continue
            init, best = snapshot(align)
            log(f"{align} levelled (vals {[(s, a) for s, a, _ in v]}; aux {[x[2] for x in v]}); best {best}")
            note = (f"stronger-branch comparison (user: 'Yes, both'): identical to {twin} (dropout 0.5, same recipe) except "
                    f"the branch, aligned from the 50-epoch Stage 1a best (step 204000, 40.9% aux val) instead of the "
                    f"10-epoch one (37.1%); init = {align} best.pt at step {best['step']} (LLM val {best.get('llm_acc')}); "
                    f"launched automatically by auto/auto_stage2.py once alignment levelled")
            r = subprocess.run([f"{A}/launch_s2wd.sh", mode, init, s2, "0.5", note],
                               capture_output=True, text=True, timeout=300)
            log(f"launch {s2}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
            if r.returncode == 0:
                set_priority(align, 1)
                log(f"{align} -> priority 1")
            todo.remove(pair)
        except Exception as e:                       # keep watching the other run
            log(f"{align}: error {e!r}")
    if todo:
        time.sleep(300)
log("done" if not todo else f"deadline reached; still waiting on {[p[0] for p in todo]}")
