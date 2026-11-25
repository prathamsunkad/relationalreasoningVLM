import json,glob,math,collections,sys
for m in ["internvl","qwen"]:
    rs=[]
    for f in sorted(glob.glob(f"{sys.argv[1]}/{m}_zeroshot/results_shard*.json")): rs+=json.load(open(f))
    n=len(rs); files=len({r["file"] for r in rs}); err=sum(1 for r in rs if r.get("pred") is None)
    c=sum(r["correct"] for r in rs); p=c/n; z=1.96
    lo=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n); hi=(p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    pos=collections.Counter(r["pred"] for r in rs); tgt=collections.Counter(r["target"] for r in rs)
    yes=sum(s>0 for r in rs for s in r["scores"])/(8*n)
    print(f"{m}: n={n} unique={files} errors={err} acc={100*p:.2f}% CI=[{100*lo:.2f},{100*hi:.2f}] yes_rate={100*yes:.1f}%")
    print("  pred pos %:", [round(100*pos[k]/n,1) for k in range(8)])
    print("  target pos %:", [round(100*tgt[k]/n,1) for k in range(8)])
