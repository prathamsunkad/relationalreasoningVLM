"""PGM rule labels for the 1,000 shared validation problems: {file: [[object, attribute, relation], ...]}.
Reads only the listed files (no directory scan)."""
import json
import sys

import numpy as np

lst, root, out = sys.argv[1:4]
names = [l.split("\t")[0].strip() for l in open(lst) if l.strip() and not l.startswith("#")]
rules = {}
for n in names:
    d = np.load(f"{root}/{n}")
    rules[n] = [[x.decode() for x in row] for row in d["relation_structure"]]
json.dump(rules, open(out, "w"))
print(f"{len(rules)} problems; rules per problem: {np.bincount([len(v) for v in rules.values()]).tolist()}")
