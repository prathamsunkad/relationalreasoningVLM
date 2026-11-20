# Python 2/3 compatible. Packs the *_test.npz files written by the I-RAVEN generator
# into one compressed npz per configuration, plus one tar.gz of the test XML annotations.
import glob, os, re, sys, tarfile
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
for cfg in sorted(os.listdir(src)):
    files = glob.glob(os.path.join(src, cfg, 'RAVEN_*_test.npz'))
    files.sort(key=lambda p: int(re.search(r'RAVEN_(\d+)_test', p).group(1)))
    if not files: continue
    ks, imgs, tgt, pred, mtgt, struct_n = [], [], [], [], [], []
    for p in files:
        d = np.load(p, allow_pickle=True)
        im = np.asarray(d['image'], dtype=np.uint8)
        assert im.shape == (16, 160, 160), (p, im.shape)
        ks.append(int(re.search(r'RAVEN_(\d+)_test', p).group(1)))
        imgs.append(im); tgt.append(int(d['target'])); pred.append(int(d['predict']))
        mtgt.append(np.asarray(d['meta_target'], dtype=np.uint8))
    out = os.path.join(dst, 'iraven_test_%s.npz' % cfg)
    np.savez_compressed(out, k=np.asarray(ks, np.int32), image=np.stack(imgs), target=np.asarray(tgt, np.int8),
                        solver_predict=np.asarray(pred, np.int8), meta_target=np.stack(mtgt))
    tar = tarfile.open(os.path.join(dst, 'iraven_test_%s_xml.tar.gz' % cfg), 'w:gz')
    for p in files:
        x = p[:-4] + '.xml'
        if os.path.exists(x): tar.add(x, arcname=os.path.join(cfg, os.path.basename(x)))
    tar.close()
    print('%s: packed %d test puzzles, target histogram %s' % (cfg, len(files), np.bincount(tgt, minlength=8).tolist()))
