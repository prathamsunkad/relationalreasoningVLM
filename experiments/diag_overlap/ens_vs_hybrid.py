"""Rebuttal check (Nov 24): hybrid (token injection) vs score ensembling of the same two separately trained models, paired on test5000.
Ensemble score = w * log_softmax(LoRA scores) + (1 - w) * log_softmax(branch scores). w: 0.5 (untuned) and a
2-fold cross-fitted w (chosen on one half of the puzzles by accuracy, applied to the other half; no puzzle is scored
with a weight it helped choose). McNemar exact test vs the hybrid on the same puzzles.

    python3 experiments/diag_overlap/ens_vs_hybrid.py results/relational/test5000   # reads <name>.json or .json.gz
"""
import gzip, json, math, os, random, sys
from math import comb
D = sys.argv[1] if len(sys.argv) > 1 else '.'
def recs(n):
    p = f'{D}/{n}.json'
    return json.load(gzip.open(p + '.gz') if not os.path.exists(p) else open(p))['records']
def load(n, key):
    return {r['id']: (r[key], r['label']) for r in recs(n) if key in r}
def lsm(v):
    m = max(v); s = math.log(sum(math.exp(x - m) for x in v)) + m; return [x - s for x in v]
def ens(L, B, w, ids):
    out = {}
    for i in ids:
        a, y = L[i]; b, _ = B[i]; la, lb = lsm(a), lsm(b)
        sc = [w * x + (1 - w) * z for x, z in zip(la, lb)]; out[i] = sc.index(max(sc)) == y
    return out
def mc(a, b):
    b10 = sum(a[k] and not b[k] for k in a); b01 = sum(b[k] and not a[k] for k in a); n = b10 + b01; k = min(b10, b01)
    return b10, b01, min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1
grid = [i / 20 for i in range(21)]
cases = [('H12 (276k branch, 5% LoRA)', 'lora5', 'br276k', ['lf276_int', 'lf276_fuse']),
         ('H14 (636k branch, 5% LoRA)', 'lora5p0', 'br636k', ['lf636_int', 'lf636_fuse']),
         ('H15 ep1 (636k branch, 20% LoRA ep1)', 'lora20e1', 'br636k', ['lf20e1_int', 'lf20e1_fuse']),
         ('H15 ep4 (636k branch, 20% LoRA ep4)', 'lora20e4', 'br636k', ['lf20e4_int', 'lf20e4_fuse'])]
for name, ln, bn, hyb in cases:
    L, B = load(ln, 'llm_scores'), load(bn, 'aux_scores'); ids = sorted(set(L) & set(B))
    e5 = ens(L, B, 0.5, ids)
    random.Random(0).shuffle(ids); h1, h2 = ids[:len(ids)//2], ids[len(ids)//2:]
    cf, ws = {}, []
    for fit, app in [(h1, h2), (h2, h1)]:
        w = max(grid, key=lambda w: (sum(ens(L, B, w, fit).values()), -abs(w - 0.5))); ws.append(w); cf.update(ens(L, B, w, app))
    acc = lambda d: 100 * sum(d.values()) / len(d)
    print(f'{name}: n={len(ids)} LoRA {acc({i: L[i][0].index(max(L[i][0])) == L[i][1] for i in ids}):.1f} branch {acc({i: B[i][0].index(max(B[i][0])) == B[i][1] for i in ids}):.1f} | ens w=.5 {acc(e5):.1f} | ens cross-fit (w={ws}) {acc(cf):.1f}')
    for h in hyb:
        H = {r['id']: r['llm_pred'] == r['label'] for r in recs(h)}
        for lab, E in [('w=.5', e5), ('cross-fit', cf)]:
            k = [i for i in ids if i in H]; a = {i: H[i] for i in k}; b = {i: E[i] for i in k}
            b10, b01, p = mc(a, b)
            print(f'   {h:12s} {acc(a):.1f} vs ens {lab:9s} {acc(b):.1f}: diff {acc(a)-acc(b):+.1f}  only-hybrid {b10} only-ens {b01} p={p:.2g}')
