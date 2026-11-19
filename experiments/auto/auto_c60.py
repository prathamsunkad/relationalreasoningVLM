"""Clean 60k hybrids (user, Nov 17: 'can you do H8, H9, H12 for the 60k model?'). Everything below sees only the
seed-0 60k PGM problems that LoRA alone 5% used.

  A. Wait for both 60k branch runs (aux_ft60k_q_L6_lr3e-4 / _lr1e-3, 40 epochs = 150k steps) to finish (final.pt), or
     to stop (chain gone 3 checks in a row). Take the better one by best.json (best validation) and copy its best.pt to
     snapshots/aux_ft60k_best.pt (diag_5k's br60 scores it alone on the test set).
  B. Launch the two 60k alignments (auto/launch_align60k.sh, interleave and fusion).
  C. When an alignment levels off (same rule as auto_lf276.py: >= 3 validations and the best did not rise by more than
     1 point over the last two, or the run finished), snapshot its best.pt and launch, for that regime:
       H8   lora_w5_q_<m>_c60   Stage 2, LoRA from scratch, no dropout      (launch_s2wd.sh, dropout 0)
       H9   lora_w5d_q_<m>_c60  Stage 2, LoRA from scratch, dropout 0.5     (launch_s2wd.sh, dropout 0.5)
       H12  lora_lf_q_<m>_c60   LoRA-first from LoRA alone 5% (step 14k)     (launch_lf2.sh)
     all with the 60k slot model and tier 2; then the alignment drops to tier 3. H12's branch-off controls are reused
     (lora_lf_q_{int,fuse}_ft50_ctrl: they never use the branch and already trained only on the 60k).
  D. Start two auto/wait_5k.sh waiters: diag_5k for the H12 runs + br60, diag_5kw for the H8/H9 runs (they take ~4x
     longer), so the faster results are not held back.
Runs detached on the login node (nohup setsid); log in logs/auto_c60_*.log. --dry_run checks inputs and prints the plan."""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

R = "/home/hice1/psunkad3/scratch/relvlm_runs"
A = f"{R}/auto"
SLOT60 = f"{R}/s0_qwen_pgm60k/best.pt"
BRANCHES = ["aux_ft60k_q_L6_lr3e-4", "aux_ft60k_q_L6_lr1e-3"]
ALIGN = {"int": "align_q_int_L6ft60k_lr3e-4", "fuse": "align_q_fuse_L6ft60k_lr3e-4"}
DRY = "--dry_run" in sys.argv
# Opt-in early set (user, Nov 17: 'can you use the current 60k branches run H8,H9 and H12 for them as well?'):
#   --branch <pt> uses that checkpoint now instead of waiting for the finished 60k branches;
#   --suffix <s> names the runs lora_*_q_<m>_<s>, align_q_<m>_L6ft<s>_lr3e-4, evals <s>_* and br<s>.
def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
FIXED_BRANCH = _arg("--branch")
SUF = _arg("--suffix", "c60")
BRNAME = "br60" if SUF == "c60" else "br" + SUF[1:]   # c60e -> br60e, matching diag_5k.sbatch (was f"br{SUF}" = brc60e, which matched nothing)
QUOTE = ("user: 'can you do H8, H9, H12 for the 60k model?'" if SUF == "c60" else
         "user: 'can you use the current 60k branches run H8,H9 and H12 for them as well?'")
if SUF != "c60":
    ALIGN = {m: f"align_q_{m}_L6ft{SUF}_lr3e-4" for m in ALIGN}
DEADLINE = time.time() + 10 * 86400
LOG = sys.stdout if DRY else open(f"{R}/logs/auto_{SUF}_{time.strftime('%m%d_%H%M%S')}.log", "a", buffering=1)
USER = os.environ.get("USER", "psunkad3")


def log(*a):
    print(time.strftime("%b %d %H:%M:%S"), *a, file=LOG)


def queue():
    return subprocess.run(["squeue", "-u", USER, "-h", "-o", "%j"], capture_output=True, text=True,
                          timeout=60).stdout.split()


def vals(tag):
    out = {}
    for f in glob.glob(f"{R}/logs/stage1_{tag}_*.log"):
        for m in re.finditer(r"^\[val\] step (\d+) pgm: llm_acc ([\d.]+) aux_acc ([\d.]+)",
                             open(f, errors="replace").read(), re.M):
            out[int(m[1])] = (float(m[2]), float(m[3]))
    return [(s, *out[s]) for s in sorted(out)]


def crashed(tag):
    fs = sorted(glob.glob(f"{R}/logs/stage1_{tag}_*.log"), key=os.path.getmtime)
    return bool(fs) and "Traceback" in open(fs[-1], errors="replace").read() and tag not in queue()


def set_priority(tag, n):
    p = f"{R}/priority.json"
    shutil.copy(p, f"{R}/" + time.strftime(".priority_backup_%m%d_%H%M%S.json"))
    d = json.load(open(p)); d[tag] = n
    json.dump(d, open(p, "w"), indent=2)
    q = subprocess.run(["squeue", "-u", USER, "-h", "-t", "PD", "-n", tag, "-o", "%i"],
                       capture_output=True, text=True, timeout=60).stdout.split()
    for j in q:
        subprocess.run(["scontrol", "update", f"JobId={j}", f"Nice={n}"], capture_output=True, timeout=60)


def snapshot(run, dst_dir, name=None):
    """Copy best.pt while making sure it was not rewritten during the copy."""
    os.makedirs(dst_dir, exist_ok=True)
    for _ in range(5):
        b1, m1 = open(f"{R}/{run}/best.json").read(), os.path.getmtime(f"{R}/{run}/best.pt")
        step = json.loads(b1)["step"]
        dst = f"{dst_dir}/{name or f'init_{run}_step{step}.pt'}"
        shutil.copy2(f"{R}/{run}/best.pt", dst)
        if open(f"{R}/{run}/best.json").read() == b1 and os.path.getmtime(f"{R}/{run}/best.pt") == m1:
            return dst, json.loads(b1)
        time.sleep(60)
    raise RuntimeError(f"{run}/best.pt kept changing")


def run(cmd, env=None):
    if DRY:
        log("DRY would run:", " ".join(cmd), "| env", env or {}); return 0, "dry", ""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**os.environ, **(env or {})})
    return r.returncode, r.stdout.strip(), r.stderr.strip()


# ---- checks
for f in [SLOT60, f"{A}/launch_align60k.sh", f"{A}/launch_s2wd.sh", f"{A}/launch_lf2.sh", f"{A}/wait_5k.sh"]:
    assert os.path.exists(f), f"missing {f}"
for t in list(ALIGN.values()) + [f"lora_{k}_q_{m}_{SUF}" for k in ("w5", "w5d", "lf") for m in ALIGN]:
    assert not os.path.exists(f"{R}/{t}"), f"{R}/{t} already exists"
log("start", "(dry run)" if DRY else "", f"suffix {SUF}", f"branch {FIXED_BRANCH or 'wait for the finished 60k branches'}")

# ---- A: wait for the 60k branch runs
miss = {b: 0 for b in BRANCHES}
while not DRY and not FIXED_BRANCH:
    q = queue()
    for b in BRANCHES:
        miss[b] = 0 if (b in q or os.path.exists(f"{R}/{b}/final.pt")) else miss[b] + 1
    if all(os.path.exists(f"{R}/{b}/final.pt") or miss[b] >= 3 for b in BRANCHES):
        break
    if time.time() > DEADLINE:
        log("deadline reached waiting for the 60k branches"); sys.exit(1)
    time.sleep(600)
time.sleep(0 if DRY else 120)
cand = []
for b in ([] if FIXED_BRANCH else BRANCHES):
    if os.path.exists(f"{R}/{b}/best.json"):
        bj = json.load(open(f"{R}/{b}/best.json"))
        cand.append((bj["aux_acc"], b, bj, os.path.exists(f"{R}/{b}/final.pt")))
        log(f"{b}: best {bj}, finished={cand[-1][3]}")
if FIXED_BRANCH:
    bj = json.load(open(FIXED_BRANCH[:-3] + ".json"))
    acc, BR, fin = bj["aux_acc"], f"{bj['source']} step {bj['step']}", False
    cand = [(acc, BR, bj, fin)]
if not cand:
    log("no 60k branch checkpoint; stopping"); sys.exit(1)
acc, BR, bj, fin = max(cand)
log(f"using {BR} (best val {acc} at step {bj['step']}, finished={fin})")
if FIXED_BRANCH:
    BR_PT = FIXED_BRANCH
elif DRY:
    BR_PT = f"{R}/{BR}/best.pt"
else:
    BR_PT, bj = snapshot(BR, f"{R}/snapshots", "aux_ft60k_best.pt")
    json.dump({"source": BR, **bj}, open(f"{R}/snapshots/aux_ft60k_best.json", "w"))
    log(f"snapshot {BR_PT}")

# ---- B: launch the two 60k alignments
for m, tag in ALIGN.items():
    note = (f"clean 60k step 3 ({QUOTE}): alignment on the 60k only from {BR} "
            f"best (aux val {acc}, step {bj['step']}); then H8/H9/H12 by auto/auto_c60.py; tier 2")
    rc, out, err = run([f"{A}/launch_align60k.sh", m, BR_PT, tag, note])
    log(f"launch {tag}: rc={rc} {out} {err}")
if DRY:
    log("DRY: would then wait for each alignment to level off and launch:",
        [f"lora_{k}_q_{m}_{SUF}" for m in ALIGN for k in ("w5", "w5d", "lf")]); sys.exit(0)

# ---- C: on levelling, launch H8, H9, H12 for that regime
todo = dict(ALIGN)
LAUNCHED = []
while todo and time.time() < DEADLINE:
    for m, align in list(todo.items()):
        try:
            v = vals(align)
            if crashed(align):
                log(f"{align} CRASHED after vals {v}; nothing launched for {m}"); todo.pop(m); continue
            fin = os.path.exists(f"{R}/{align}/final.pt")
            lev = len(v) >= 3 and max(x[1] for x in v[-2:]) - max(x[1] for x in v[:-2]) <= 0.01
            if not (lev or (fin and v)):
                continue
            init, best = snapshot(align, f"{A}/snap")
            log(f"{align} levelled (vals {[(s, a) for s, a, _ in v]}; aux {[x[2] for x in v]}); best {best}")
            base = (f"clean 60k ({QUOTE}): branch {BR} trained only on the 60k, "
                    f"aligned on the 60k ({align} best step {best['step']}, LLM val {best.get('llm_acc')}); "
                    f"slots s0_qwen_pgm60k; tier 2; launched by auto/auto_c60.py")
            env = {"SLOT_CKPT": SLOT60, "PRIO": "1"}
            for tag, cmd in [
                (f"lora_w5_q_{m}_{SUF}", [f"{A}/launch_s2wd.sh", m, init, f"lora_w5_q_{m}_{SUF}", "0",
                                        f"H8 recipe (Stage 2, no dropout, = lora_w5_q_{m}_L6ft); {base}"]),
                (f"lora_w5d_q_{m}_{SUF}", [f"{A}/launch_s2wd.sh", m, init, f"lora_w5d_q_{m}_{SUF}", "0.5",
                                         f"H9 recipe (Stage 2, dropout 0.5, = lora_w5d_q_{m}_L6ft); {base}"]),
                (f"lora_lf_q_{m}_{SUF}", [f"{A}/launch_lf2.sh", m, init, f"lora_lf_q_{m}_{SUF}",
                                        f"H12 recipe (LoRA-first, = lora_lf_q_{m}_ft276k); control lora_lf_q_{m}_ft50_ctrl "
                                        f"(branch off, reused); {base}"])]:
                rc, out, err = run(cmd, env)
                log(f"launch {tag}: rc={rc} {out} {err}")
                if rc == 0:
                    LAUNCHED.append(tag)
            set_priority(align, 2)
            log(f"{align} -> priority 2")
            todo.pop(m)
        except Exception as e:
            log(f"{align}: error {e!r}")
    if todo:
        time.sleep(300)

# ---- D: test scoring
EV = {}
for m in ALIGN:
    EV[f"lora_lf_q_{m}_{SUF}"] = f"{SUF}_lf_{m},{SUF}_lf_{m}_off"
    EV[f"lora_w5_q_{m}_{SUF}"] = f"{SUF}_w5_{m},{SUF}_w5_{m}_off"
    EV[f"lora_w5d_q_{m}_{SUF}"] = f"{SUF}_w5d_{m},{SUF}_w5d_{m}_off"
fast = [t for t in LAUNCHED if t.startswith("lora_lf")]
slow = [t for t in LAUNCHED if not t.startswith("lora_lf")]
for job, tl, tags, extra in [("diag_5k", "02:30:00", fast, [f"-={BRNAME}"]), ("diag_5kw", "04:00:00", slow, [])]:
    if not tags and not extra:
        continue
    cmd = [f"{A}/wait_5k.sh", job, tl] + [f"{t}={EV[t]}" for t in tags] + extra
    subprocess.Popen(["nohup", "setsid"] + cmd, stdout=open(f"{A}/wait_5k_{job}.log", "a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL)
    log("started", " ".join(cmd))
log("done" if not todo else f"deadline reached; still waiting on {list(todo.values())}")
