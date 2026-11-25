"""Strict adapter for the shared Slot Abstractor package (weights stay external).

Load the supplied reference source, not a reimplementation of its reasoner.
The only source portability adjustment is build_grid's device allocation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def checked(path, expected):
    actual=sha256(path)
    if actual!=expected: raise ValueError('Checksum mismatch: '+str(path))
    return actual


def pool_relations(states):
    """Post-reasoning slot means per panel; keeps candidate and panel axes."""
    if states.ndim!=4 or tuple(states.shape[1:])!=(8,144,32):
        raise ValueError('Expected reference 8 x 144 x 32 relational states')
    return states.reshape(states.shape[0],8,9,16,32).mean(3)


def question_seed(identifier,seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{identifier}'.encode()).digest()[:8],'big')%(2**63-1)


class FrozenRelations:
    def __init__(self, package, hashes, device='cuda'):
        root=Path(package)/'pgm'
        for rel,digest in hashes.items(): checked(root/rel,digest)
        self.device=torch.device(device)
        self.root=root
        self.layout=json.loads((root/'layout.json').read_text())
        # Import reference module and its local util without permanently changing sys.path/modules.
        previous_util=sys.modules.pop('util',None)
        sys.path.insert(0,str(root))
        try:
            spec=importlib.util.spec_from_file_location('_shared_pgm_reference',root/'slot_abstractor_pgm.py')
            source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
        finally:
            sys.path.pop(0);sys.modules.pop('util',None)
            if previous_util is not None: sys.modules['util']=previous_util
        def grid(resolution):
            ranges=[np.linspace(0.,1.,num=res) for res in resolution]
            value=np.stack(np.meshgrid(*ranges,sparse=False,indexing='ij'),axis=-1)
            value=value.reshape(1,resolution[0],resolution[1],-1).astype(np.float32)
            return torch.from_numpy(np.concatenate([value,1.-value],axis=-1)).to(self.device)
        source.build_grid=grid
        self.source=source
        ae=torch.load(root/'weights/slot_ae_pgm20.pth.tar',map_location='cpu',weights_only=True)
        reason=torch.load(root/'weights/slot_abstractor_pgm_pgm20_1gpu_d24_lr8e-05_run0_best.pth.tar',
                          map_location='cpu',weights_only=True)
        self.encoder=source.SlotAttentionAutoEncoder((80,80),16,3,32).to(self.device)
        self.scorer=source.scoring_model(SimpleNamespace(apply_context_norm=True),32,24,8,512,16).to(self.device)
        strip=lambda d:{k.removeprefix('module.'):v for k,v in d.items()}
        self.encoder.load_state_dict(strip(ae['slot_model_state_dict']),strict=True)
        self.scorer.load_state_dict(strip(reason['abstractor_model_state_dict']),strict=True)
        for m in (self.encoder,self.scorer): m.eval().requires_grad_(False)
        self.parity_verified=False

    def panels(self,image):
        from torchvision import transforms as T
        from pgmvlm.panels import split_composite
        if list(image.size)!=self.layout['sheet_size']: raise ValueError('Wrong contact sheet')
        boxes=[b for i,b in enumerate(self.layout['matrix']) if i!=self.layout['missing']]+self.layout['choices']
        gray=image.convert('RGB').convert('L')
        transform=T.Compose([T.ToTensor(),T.Lambda(lambda x:2*x-1.),T.Resize((80,80))])
        # Independently cross-check canonical crops against supplied layout.
        canonical=split_composite(image)
        panels=[]
        for panel,box in zip(canonical,boxes):
            crop=gray.crop(tuple(box)).resize((160,160),Image.Resampling.BICUBIC)
            if not np.array_equal(np.asarray(panel.convert('L')),np.asarray(crop)):
                raise ValueError('Reference layout and canonical panel crops disagree')
            panels.append(transform(crop))
        return torch.stack(panels).to(self.device)

    def encode_panel(self,panel,reference=False):
        if reference:
            result=self.encoder(panel,self.device)
            return result[3],result[4]
        # Exact feature prefix of reference forward; bypass unused reconstruction.
        x,features,positions=self.encoder.encoder_cnn(panel)
        x=F.layer_norm(x,x.shape[1:],torch.ones(x.shape[1:],device=x.device),torch.zeros(x.shape[1:],device=x.device))
        x=self.encoder.fc2(F.relu(self.encoder.fc1(x)))
        _,feat,pos,_=self.encoder.slot_attention(x,features,positions,self.device)
        return feat,pos

    def score_features(self,features,positions):
        outputs=[]
        hook=self.scorer.abstractors.abstractor.register_forward_hook(lambda m,a,out:outputs.append(out))
        try:
            scores=self.scorer(features[:,:8],positions[:,:8],features[:,8:],positions[:,8:],self.device)
        finally: hook.remove()
        states=torch.stack(outputs,dim=1)
        bank=pool_relations(states)
        reconstructed=self.scorer.abstractors.mlp_head(bank.mean(2)).squeeze(-1)
        torch.testing.assert_close(scores,reconstructed,rtol=1e-5,atol=1e-5)
        return bank,scores

    @torch.no_grad()
    def extract(self,image,identifier,seed=20261123,verify=False):
        panels=self.panels(image)
        features,positions=[],[]
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            for i,panel in enumerate(panels):
                value=question_seed(f'{identifier}:panel:{i}',seed)
                torch.manual_seed(value)
                if self.device.type=='cuda': torch.cuda.manual_seed(value)
                feat,pos=self.encode_panel(panel[None],reference=False)
                if verify:
                    torch.manual_seed(value)
                    if self.device.type=='cuda': torch.cuda.manual_seed(value)
                    ref_feat,ref_pos=self.encode_panel(panel[None],reference=True)
                    torch.testing.assert_close(feat,ref_feat,rtol=0,atol=0)
                    torch.testing.assert_close(pos,ref_pos,rtol=0,atol=0)
                features.append(feat);positions.append(pos)
            features=torch.stack(features,1);positions=torch.stack(positions,1)
            bank,scores=self.score_features(features,positions)
            if verify:
                permutation=[3,7,0,6,2,1,5,4]
                order=list(range(8))+[8+p for p in permutation]
                reordered,reordered_scores=self.score_features(features[:,order],positions[:,order])
                torch.testing.assert_close(bank[:,permutation],reordered,rtol=0,atol=1e-5)
                torch.testing.assert_close(scores[:,permutation],reordered_scores,rtol=0,atol=1e-5)
                self.parity_verified=True
        if not self.parity_verified: raise RuntimeError('Reference parity must pass before caching')
        if not torch.isfinite(bank).all(): raise RuntimeError('Nonfinite teacher features')
        return bank.detach().cpu()
