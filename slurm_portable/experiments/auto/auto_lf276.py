# portable copy (tools/make_portable.py): paths from env.sh (source it first)
"""LoRA-first with the step-276k branch (user, Nov 16: 'Yes, queue it'): wait until each alignment run of the
276k branch levels off, then launch its LoRA-first run (auto/launch_lf2.sh, the lora_lf_q_*_ft50 recipe) from the
alignment's best.pt. Same levelling rule as auto_stage2.py: at least 3 validations and the best did not rise by more
than 1 point over the last two (or the run finished). After launching, the alignment drops to priority 2. When all
are launched, starts auto/wait_5k.sh so the finished runs are scored on the 5,000 test problems.
Runs detached on the login node (nohup setsid); log in logs/auto_lf276_*.log."""
import glob
import json
import os
import re
import shutil
import subprocess
import time

R = os.environ["RELVLM_RUNS"]
A = os.environ["REPO_ROOT"] + "/slurm_portable/experiments/auto"   # the portable launchers
PAIRS = [("align_q_int_L6ft276k_lr3e-4", "int", "lora_lf_q_int_ft276k", "lora_lf_q_int_ft50"),
         ("align_q_fuse_L6ft276k_lr3e-4", "fuse", "lora_lf_q_fuse_ft276k", "lora_lf_q_fuse_ft50")]
DEADLINE = time.time() + 4 * 86400
LOG = open(f"{R}/logs/auto_lf276_{time.strftime('%m%d_%H%M%S')}.log", "a", buffering=1)


def log(*a):
    print(time.strftime("%b %d %H:%M:%S"), *a, file=LOG)


def vals(tag):
    out = {}
    for f in glob.glob(f"{R}/logs/stage1_{tag}_*.log"):
        for m in re.finditer(r"^\[val\] step (\d+) pgm: llm_acc ([\d.]+) aux_acc ([\d.]+)", open(f, errors="replace").read(), re.M):
            out[int(m[1])] = (float(m[2]), float(m[3]))
    return [(s, *out[s]) for s in sorted(out)]


def in_queue(tag):
    q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"],
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
    q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-t", "PD", "-n", tag, "-o", "%i"],
                       capture_output=True, text=True, timeout=60).stdout.split()
    for j in q:
        subprocess.run(["scontrol", "update", f"JobId={j}", f"Nice={n}"], capture_output=True, timeout=60)


def snapshot(align):
    """Copy best.pt while making sure it was not rewritten during the copy."""
    os.makedirs(f"{R}/auto/snap", exist_ok=True)
    for _ in range(5):
        b1, m1 = open(f"{R}/{align}/best.json").read(), os.path.getmtime(f"{R}/{align}/best.pt")
        step = json.loads(b1)["step"]
        dst = f"{R}/auto/snap/init_{align}_step{step}.pt"
        shutil.copy2(f"{R}/{align}/best.pt", dst)
        if open(f"{R}/{align}/best.json").read() == b1 and os.path.getmtime(f"{R}/{align}/best.pt") == m1:
            return dst, json.loads(b1)
        time.sleep(60)
    raise RuntimeError(f"{align}/best.pt kept changing")


todo = list(PAIRS)
LAUNCHED = []
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
            note = (f"LoRA-first with the stronger branch (user: 'Yes, queue it'): identical to {twin} except the branch, "
                    f"aligned from the 50-epoch run's step-276k snapshot (47.6% aux val, 45.0% test) instead of step 204k; "
                    f"init = {align} best.pt at step {best['step']} (LLM val {best.get('llm_acc')}); control = "
                    f"{twin}_ctrl (branch off, does not depend on the branch); tier 2; launched by auto/auto_lf276.py")
            r = subprocess.run([f"{A}/launch_lf2.sh", mode, init, s2, note],
                               capture_output=True, text=True, timeout=300)
            log(f"launch {s2}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
            if r.returncode == 0:
                set_priority(align, 2)
                log(f"{align} -> priority 2")
                LAUNCHED.append(s2)
            todo.remove(pair)
        except Exception as e:                       # keep watching the other run
            log(f"{align}: error {e!r}")
    if todo:
        time.sleep(300)
if LAUNCHED:
    ev = {"lora_lf_q_int_ft276k": "lf276_int,lf276_int_off", "lora_lf_q_fuse_ft276k": "lf276_fuse,lf276_fuse_off"}
    cmd = [f"{A}/wait_5k.sh", "diag_5k", "01:45:00"] + [f"{t}={ev[t]}" for t in LAUNCHED]
    subprocess.Popen(["nohup", "setsid"] + cmd, stdout=open(f"{R}/auto/wait_5k_diag_5k.log", "a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL)
    log("started", " ".join(cmd))
log("done" if not todo else f"deadline reached; still waiting on {[p[0] for p in todo]}")
