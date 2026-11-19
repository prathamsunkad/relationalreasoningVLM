"""Regenerate slurm_portable/: copies of every Slurm / launcher script with no account-specific paths.

    python3 tools/make_portable.py

Each copy takes its paths from env.sh (edit that one file), writes Slurm logs to <repo>/logs/, and chains or
submits other portable copies, never the originals. The originals stay as they ran, as a record.
"""
import os
import re
import stat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/slurm_portable"
H = "/home/hice1/psunkad3"
S = f"{H}/scratch"
SOURCES = ["slurm", "relational/slurm", "zeroshot/slurm", "experiments/auto", "experiments/diag_5k",
           "experiments/diag_5kw", "experiments/diag_overlap", "experiments/diag_branchoff", "share", "tools"]
LOADER = ('# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/\n'
          'source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || '
          '{ echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }\n'
          'mkdir -p "$REPO_ROOT/logs"\n')


def group(rel):
    return "relational" if rel.startswith(("relational/", "experiments/")) else "baseline"


def convert_shell(text, rel):
    g = group(rel)
    # 1. the REPO / RUNS variables mean different folders in the two code bases
    if g == "relational":
        text = re.sub(r"REPO=\$\{REPO:-" + S + r"/relvlm(_ft)?\}", "REPO=$RELVLM_REPO", text)
        text = re.sub(r"REPO=" + S + r"/relvlm(_ft)?\b", "REPO=$RELVLM_REPO", text)
        text = re.sub(r"RUNS=\$\{RUNS:-" + S + r"/relvlm_runs\}", "RUNS=$RELVLM_RUNS", text)
        text = re.sub(r"RUNS=" + S + r"/relvlm_runs\b", "RUNS=$RELVLM_RUNS", text)
    else:
        text = text.replace("REPO=${REPO:-" + S + "/pgm-vlm-reasoning}", "REPO=$REPO_ROOT")
        text = text.replace("RUNS=${RUNS:-" + S + "/vlm_lora_pc_runs}", "RUNS=${RUNS:-$LORA_RUNS}")
    # 2. path prefixes, longest first
    for old, new in [
        (f"{H}/.cache/huggingface", "$HF_HOME"),
        (f"{S}/vlm_lora_pc_runs/filecache/neutral/val1000_shared.tsv", "$REPO_ROOT/results/splits/val1000_shared.tsv"),
        (f"{S}/vlm_lora_pc_runs/qwen_pc_neutral5pct_recipe/best", "$LORA_ADAPTER_5PCT"),
        (f"{S}/vlm_lora_pc_runs", "$LORA_RUNS"),
        (f"{S}/relvlm_runs", "$RELVLM_RUNS"),
        (f"{S}/relvlm_data", "$RELVLM_DATA"),
        (f"{S}/pgm-vlm-reasoning/env.sh", "$REPO_ROOT/env.sh"),
        (f"{S}/pgm-vlm-reasoning", "$REPO_ROOT"),
        (f"{S}/pgm_data", "$DATA"),
        (f"{S}/clevr_art_data", "$CLEVR_ART_DATA"),
        (f"{H}/pgm20", "$WORK/pgm20"),
    ]:
        text = text.replace(old, new)
    text = re.sub(S + r"/relvlm(_ft)?\b", "$RELVLM_REPO", text)
    text = text.replace(S + "/", "$WORK/")
    # 3. #SBATCH -o/-e cannot expand variables: logs/ relative to the submit directory (the repo root)
    text = re.sub(r"^(#SBATCH\s+-[oe]\s+)\S*/([^/\s]+)$", r"\1logs/\2", text, flags=re.M)
    # 4. chains and submissions go to the portable copies; resubmitted segments log to the repo's logs/
    if g == "relational":
        text = text.replace("$REPO/slurm/", "$REPO_ROOT/slurm_portable/relational/slurm/")
        text = re.sub(r'"\$D/(diag_5k\.sbatch)"', r'"$REPO_ROOT/slurm_portable/experiments/diag_5k/\1"', text)
        text = re.sub(r"\$RELVLM_RUNS/auto/", "$REPO_ROOT/slurm_portable/experiments/auto/", text)
    else:
        text = text.replace("$REPO/slurm/", "$REPO_ROOT/slurm_portable/slurm/")
    text = re.sub(r"sbatch --export=ALL \$\{SB_OPTS:-\}", 'sbatch --export=ALL -o "$REPO_ROOT/logs/%x_%j.log" ${SB_OPTS:-}', text)
    # scripts called by relative path or from the runs folder: point at the portable copies
    if g == "relational":
        text = text.replace("--export=ALL slurm/tests.sbatch", "--export=ALL $REPO_ROOT/slurm_portable/relational/slurm/tests.sbatch")
        text = text.replace("A=$RELVLM_RUNS/auto", "A=$REPO_ROOT/slurm_portable/experiments/auto").replace("A=$RUNS/auto", "A=$REPO_ROOT/slurm_portable/experiments/auto")
    else:
        text = re.sub(r"(^|\s)slurm/(eval_array|train|tests|overfit|viz)\.sbatch", r"\1$REPO_ROOT/slurm_portable/slurm/\2.sbatch", text, flags=re.M)
    # python heredocs inside the launchers do not expand $VARS: read the path from the environment
    text = text.replace('p = "$RELVLM_RUNS/priority.json"', 'p = os.environ["RELVLM_RUNS"] + "/priority.json"')
    text = text.replace('import json, sys, shutil, time', 'import json, os, sys, shutil, time')
    # 5. load env.sh right after the #! / #SBATCH header
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    loader = LOADER
    if rel.startswith("experiments/auto/"):   # bookkeeping files the launchers append to: create them on a fresh account
        loader += ('mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"\n'
                   '[ -f "$RELVLM_RUNS/priority.json" ] || echo \'{"_about": ""}\' > "$RELVLM_RUNS/priority.json"\n')
    lines.insert(i, loader)
    return "\n".join(lines)


def convert_python(text):
    if "import os" not in text:
        text = text.replace("import ", "import os\nimport ", 1)
    text = text.replace(f'"{S}/relvlm_runs"', 'os.environ["RELVLM_RUNS"]')
    text = text.replace('{A}/snap', '{R}/auto/snap').replace('{A}/wait_5k_', '{R}/auto/wait_5k_')
    text = text.replace('A = f"{R}/auto"', 'A = os.environ["REPO_ROOT"] + "/slurm_portable/experiments/auto"   # the portable launchers')
    text = text.replace(f"{S}/relvlm_runs", '" + os.environ["RELVLM_RUNS"] + "')
    text = text.replace('os.environ.get("USER", "psunkad3")', 'os.environ["USER"]')
    return ('# portable copy (tools/make_portable.py): paths from env.sh (source it first)\n' + text) if not text.startswith("#!") else text


n = 0
for src in SOURCES:
    d = f"{ROOT}/{src}"
    for f in sorted(os.listdir(d)):
        p = f"{d}/{f}"
        if not os.path.isfile(p) or f.startswith(".") or not f.endswith((".sbatch", ".sh", ".py")):
            continue
        if src == "tools" and not f.endswith(".sbatch"):
            continue
        rel = f"{src}/{f}"
        text = open(p).read()
        if f.endswith(".py"):
            if src != "experiments/auto" or f == "ptshapes.py":
                continue
            new = convert_python(text)
        else:
            if f in ("plots.sbatch", "unpack_results.sh"):        # already path-free
                continue
            new = convert_shell(text, rel)
        o = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(o), exist_ok=True)
        open(o, "w").write(new)
        os.chmod(o, os.stat(p).st_mode | stat.S_IXUSR)
        n += 1
print(n, "portable scripts in", OUT)
