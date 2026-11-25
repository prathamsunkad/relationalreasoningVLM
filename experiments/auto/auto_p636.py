"""Phases 0-2 with the 50-epoch branch at step 636k (user, Nov 23: 'Yes, start phase 0', then 'start all the phases and
prioritize the phases accordingly'). Runs detached on the login node; log in logs/auto_p636_*.log.
  1. Phase 0 analysis: once diag_5k has br636k, lora20e{1,2,4}, lora5p0, run diag_overlap/analyze.py on diag_p0/
     (LoRA x branch overlap, averaging, per rule) -> diag_p0/report.md.
  2. When align_q_{int,fuse}_L6ft636k_lr3e-4 levels off (>= 3 validations, best up <= 1 point over the last two, or
     finished), from its best.pt launch: phase 1 LoRA-first at 5% (launch_lf2.sh, tier 1) and Stage 2 dropout 0.5
     (launch_s2wd.sh, tier 2); phase 2 LoRA-first on the 20% LoRA after epochs 1 and 4 (launch_lf20.sh, tier 1;
     controls lora_lf20e*_q_*_ctrl were launched by hand). The alignment then drops to tier 2.
  3. Test scoring as groups finish (auto/wait_5k.sh -> diag_5k.sbatch, names defined there), and A3 token ablations
     (diag_ablate/submit_ablate.sh) for the interleave hybrids and the interleave alignment."""
import glob, json, os, re, shutil, subprocess, time

R = "/home/hice1/psunkad3/scratch/relvlm_runs"
A = f"{R}/auto"
LP = "/home/hice1/psunkad3/scratch/vlm_lora_pc_runs/qwen_pc_recipe_e1"
DEADLINE = time.time() + 7 * 86400
LOG = open(f"{R}/logs/auto_p636_{time.strftime('%m%d_%H%M%S')}.log", "a", buffering=1)
SRC = "50-epoch branch at step 636k (63.9% aux val)"
USER = os.environ.get("USER", "psunkad3")


def log(*a):
    print(time.strftime("%b %d %H:%M:%S"), *a, file=LOG)


def sh(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**os.environ, **(env or {})})
    log(f"{os.path.basename(cmd[0])} {cmd[1:4]}: rc={r.returncode} {r.stdout.strip()[-300:]} {r.stderr.strip()[-300:]}")
    return r.returncode == 0


def vals(tag):
    out = {}
    for f in glob.glob(f"{R}/logs/stage1_{tag}_*.log"):
        for m in re.finditer(r"^\[val\] step (\d+) pgm: llm_acc ([\d.]+) aux_acc ([\d.]+)", open(f, errors="replace").read(), re.M):
            out[int(m[1])] = (float(m[2]), float(m[3]))
    return [(s, *out[s]) for s in sorted(out)]


def in_queue(tag):
    q = subprocess.run(["squeue", "-u", USER, "-h", "-o", "%j"], capture_output=True, text=True, timeout=60).stdout.split()
    return tag in q


def set_priority(tag, n):
    p = f"{R}/priority.json"
    shutil.copy(p, f"{R}/" + time.strftime(".priority_backup_%m%d_%H%M%S.json"))
    d = json.load(open(p)); d[tag] = n; json.dump(d, open(p, "w"), indent=2)
    for j in subprocess.run(["squeue", "-u", USER, "-h", "-t", "PD", "-n", tag, "-o", "%i"],
                            capture_output=True, text=True, timeout=60).stdout.split():
        subprocess.run(["scontrol", "update", f"JobId={j}", f"Nice={n}"], capture_output=True, timeout=60)


def snapshot(align):
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


def wait5k(job, tl, specs):
    cmd = [f"{A}/wait_5k.sh", job, tl] + specs
    subprocess.Popen(["nohup", "setsid"] + cmd, stdout=open(f"{A}/wait_5k_{job}.log", "a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL)
    log("started", " ".join(cmd))


P0 = ["br636k", "lora20e1", "lora20e2", "lora20e4", "lora5p0"]
p0_done = False
aligns = {"int": "align_q_int_L6ft636k_lr3e-4", "fuse": "align_q_fuse_L6ft636k_lr3e-4"}
todo = dict(aligns)
groups = {}                      # name -> (wait_5k job, spec list)
abl_wait = {"align_q_int_L6ft636k_lr3e-4"}
abl_done = set()
log("start; phase 0 evals", P0, "alignments", list(aligns.values()))
while time.time() < DEADLINE:
    try:
        # 1. phase 0 analysis
        if not p0_done and all(os.path.getsize(f"{R}/diag_5k/{n}.json") > 0 if os.path.exists(f"{R}/diag_5k/{n}.json") else False for n in P0):
            d = f"{R}/diag_p0"; os.makedirs(d, exist_ok=True)
            for n in P0 + ["rules"]:
                if not os.path.exists(f"{d}/{n}.json"):
                    os.symlink(f"{R}/diag_5k/{n}.json", f"{d}/{n}.json")
            r = subprocess.run(["python3", f"{R}/diag_overlap/analyze.py", "--dir", d], capture_output=True, text=True, timeout=1200)
            log(f"phase 0 analysis rc={r.returncode} -> {d}/report.md {r.stderr.strip()[-300:]}")
            p0_done = True
        # 2. launches on level-off
        for mode, al in list(todo.items()):
            v = vals(al)
            fin = os.path.exists(f"{R}/{al}/final.pt")
            if not v and not in_queue(al) and fin is False and glob.glob(f"{R}/logs/stage1_{al}_*.log"):
                pass
            lev = len(v) >= 3 and max(x[1] for x in v[-2:]) - max(x[1] for x in v[:-2]) <= 0.01
            if not (lev or (fin and v)):
                continue
            init, best = snapshot(al)
            log(f"{al} levelled (vals {[(s, a) for s, a, _ in v]}; aux {[x[2] for x in v]}); best {best}")
            base = (f"(user: 'start all the phases and prioritize the phases accordingly'): {SRC}; init = {al} best.pt at "
                    f"step {best['step']} (LLM val {best.get('llm_acc')}); launched by auto/auto_p636.py")
            lf = f"lora_lf_q_{mode}_ft636k"
            if sh([f"{A}/launch_lf2.sh", mode, init, lf, f"phase 1, H12 recipe at 5% {base}; control lora_lf_q_{mode}_ft50_ctrl; tier 1"], {"PRIO": "1"}):
                groups.setdefault("g1", []).append(f"{lf}=lf636_{mode},lf636_{mode}_off")
            s2 = f"lora_w5d_q_{mode}_L6ft636k"
            if sh([f"{A}/launch_s2wd.sh", mode, init, s2, "0.5", f"phase 1, H9/H10 recipe (Stage 2, dropout 0.5) at 5% {base}; tier 2"], {"PRIO": "2"}):
                groups.setdefault("g4", []).append(f"{s2}=w5d636_{mode},w5d636_{mode}_off")
            for e in ("e1", "e4"):
                t = f"lora_lf20{e}_q_{mode}_ft636k"
                if sh([f"{A}/launch_lf20.sh", mode, init, f"{LP}/final_{e}", t, "0",
                       f"phase 2, H12 recipe on the 20% LoRA final_{e} with all 240k puzzles (matched data) {base}; control lora_lf20{e}_q_{mode}_ctrl; tier 1"], {"PRIO": "1"}):
                    groups.setdefault(f"g{e}", []).append(f"{t}=lf20{e}_{mode},lf20{e}_{mode}_off")
                    groups[f"g{e}"].append(f"lora_lf20{e}_q_{mode}_ctrl=lf20{e}_{mode}_ctrl")
            if mode == "int":
                abl_wait |= {lf, s2, "lora_lf20e1_q_int_ft636k", "lora_lf20e4_q_int_ft636k"}
            set_priority(al, 2); log(f"{al} -> priority 2")
            del todo[mode]
            if not todo:   # both launched: one scoring watcher per group (each submits when its runs have final.pt)
                for g, specs in groups.items():
                    wait5k("diag_5kw" if g == "g4" else "diag_5k", "04:00:00", specs)
        # 3. A3 ablations for finished interleave runs
        ready = [r for r in abl_wait - abl_done if os.path.exists(f"{R}/{r}/final.pt") and os.path.exists(f"{R}/{r}/best.pt")]
        if ready:
            if sh(["bash", f"{R}/diag_ablate/submit_ablate.sh"] + ready, {"TIER": "1"}):
                abl_done |= set(ready)
        if not todo and p0_done and abl_done >= abl_wait:
            break
    except Exception as e:
        log(f"error {e!r}")
    time.sleep(300)
log("done" if not todo and abl_done >= abl_wait else f"deadline/exit; todo {list(todo)}, ablations left {sorted(abl_wait - abl_done)}")
