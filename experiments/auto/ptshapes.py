"""Count parameters per module group in a torch checkpoint without torch (reads the pickled shapes). Usage: python3 ptshapes.py ckpt.pt"""
import sys, zipfile, pickle, collections, math
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name.startswith('_rebuild_tensor'):
            return lambda storage, offset, size, *a, **k: ('T', tuple(size))
        if name == '_rebuild_parameter':
            return lambda t, *a, **k: t
        if mod.startswith(('torch', 'numpy', 'collections')):
            return collections.OrderedDict if name == 'OrderedDict' else (lambda *a, **k: ('X', name))
        return super().find_class(mod, name)
    def persistent_load(self, pid): return None
z = zipfile.ZipFile(sys.argv[1]); n = [x for x in z.namelist() if x.endswith('data.pkl')][0]
obj = U(z.open(n)).load()
def walk(o, p=''):
    if isinstance(o, tuple) and len(o) == 2 and o[0] == 'T': yield p, o[1]
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f'{p}.{k}' if p else str(k))
g = collections.Counter()
for k, s in walk(obj):
    if not k.startswith('optim'): g['.'.join(k.split('.')[:2])] += math.prod(s) if s else 1
for k, c in sorted(g.items()): print(f'{c/1e6:9.3f}M  {k}')
