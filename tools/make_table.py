"""Build the paper-style PGM table (Table 4 of the Slot Abstractors paper) from merged eval directories.

  python tools/make_table.py \
      --row "Qwen3-VL-2B zero-shot:neutral=results/qwen_zs_neutral" \
      --row "Qwen3-VL-2B LoRA 100k:neutral=results/qwen_neutral,interpolation=results/qwen_interp,extrapolation=results/qwen_extrap" \
      --row "InternVL3.5-2B LoRA 100k:neutral=results/ivl_neutral"
Each dir must contain results_shard*.json (summary.json is (re)built if missing). Missing regimes print '-'.
Paper rows are copied from Table 4 (test accuracy, 1 model per regime) so you can compare side by side.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora.eval_vlm import merge  # noqa: E402

REGIMES = ["neutral", "interpolation", "hoap", "hotp", "hotriples", "holt", "hosc", "extrapolation"]
HEADERS = ["Neutral", "Intp.", "H.O.A.P.", "H.O.T.P.", "H.O.Triples", "H.O.L-T", "H.O.S-C", "Extp."]
ALIASES = {"intp": "interpolation", "interp": "interpolation", "extp": "extrapolation", "extrap": "extrapolation", "ho_ap": "hoap", "ho_tp": "hotp", "ho_triples": "hotriples",
           "ho_lt": "holt", "ho_sc": "hosc"}
PAPER = {  # Table 4, "Results reflect test accuracy for 1 trained model for each regime"
    "[paper] Slot Abstractor": [91.5, 91.6, 63.3, 78.3, 20.4, 16.7, 14.3, 39.3],
    "[paper] PredRNet": [97.4, 70.5, 63.4, 67.8, 23.4, 27.3, 13.1, 19.7],
    "[paper] ARII": [88.0, 72.0, 50.0, 64.1, 32.1, 16.0, 12.7, 29.0],
    "[paper] MRNet": [93.4, 68.1, 38.4, 55.3, 25.9, 30.1, 16.9, 19.2],
    "[paper] STSN": [98.2, 78.5, None, None, None, None, None, 20.4],
}


def load_summary(d):
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        if not os.path.isdir(d) or not any(f.startswith("results_shard") for f in os.listdir(d)):
            raise SystemExit(f"no results_shard*.json in {d}")
        merge(d)
    return json.load(open(p))


def table5(args):
    lines = ["| Model | 20% training data | 100% training data |", "|---|---|---|"]
    for row in args.row:
        label, spec = row.split(":", 1)
        cells = {"20": "-", "100": "-"}
        for item in spec.split(","):
            k, d = item.split("=", 1)
            if k not in cells:
                raise SystemExit(f"table5 keys are 20 and 100, got {k!r}")
            m = load_summary(d).get(args.metric)
            if m:
                half = 100 * (m["ci95"][1] - m["ci95"][0]) / 2
                cells[k] = f"{100*m['acc']:.1f} ±{half:.1f} (n={m['n']})"
        lines.append(f"| {label} | {cells['20']} | {cells['100']} |")
    if not args.no_paper:
        lines += ["| [paper] STSN | 53.35 | 98.2 |", "| [paper] Slot Abstractor | 71.25 | 91.5 |"]
    print("\n".join(lines))
    print(f"\nPGM neutral regime, metric: {args.metric}. Paper Table 5: test accuracy of 1 trained model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True, help='"label:regime=dir,regime=dir"')
    ap.add_argument("--metric", default="scores", choices=["scores"])
    ap.add_argument("--no_paper", action="store_true")
    ap.add_argument("--table5", action="store_true",
                    help='paper Table 5 layout (neutral regime, 20%% vs 100%% of the training data); rows "label:20=dir,100=dir"')
    args = ap.parse_args()
    if args.table5:
        return table5(args)

    lines = ["| Model | " + " | ".join(HEADERS) + " |", "|---|" + "---|" * len(HEADERS)]
    for row in args.row:
        label, spec = row.split(":", 1)
        cells = {r: "-" for r in REGIMES}
        for item in spec.split(","):
            reg, d = item.split("=", 1)
            reg = ALIASES.get(reg.lower(), reg.lower())
            if reg not in REGIMES:
                raise SystemExit(f"unknown regime {reg!r}; use one of {REGIMES} (or aliases {sorted(ALIASES)})")
            s = load_summary(d)
            m = s.get(args.metric)
            if m is None:
                continue
            half = 100 * (m["ci95"][1] - m["ci95"][0]) / 2
            flag = "" if not s.get("expected_n") or s.get("complete") else " (incomplete)"
            cells[reg] = f"{100*m['acc']:.1f} ±{half:.1f} (n={m['n']}){flag}"
        lines.append(f"| {label} | " + " | ".join(cells[r] for r in REGIMES) + " |")
    lines.append("| chance | " + " | ".join(["12.5"] * len(REGIMES)) + " |")
    if not args.no_paper:
        for name, vals in PAPER.items():
            lines.append(f"| {name} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in vals) + " |")
    print("\n".join(lines))
    print(f"\nmetric: {args.metric}; ± is the half-width of the 95% Wilson interval (binomial sampling error on the test "
          f"subset only, not seed-to-seed variance).")


if __name__ == "__main__":
    main()
