"""Case A: seven frozen-LFM interventions with addressed groups and bounded fusion.

Candidate-scoring PREFILL ONLY. One completed 520px PGM matrix, encoded as one
whole-image tile. No learned slots, pretrained branch, LoRA or answer tokens.
The historical extension implementations/checkpoint formats remain unchanged.
"""
from __future__ import annotations

import math
from types import MethodType
import torch
from torch import nn
from .abstractor_plugin import AbstractorLayer
from .relational import Attention, ordered_pool, prefix_end

DESIGNS = ('early', 'late', 'joint', 'dual', 'memory_projected', 'rca_plugin', 'memory_contextual')


def panel_groups(spatial_shape, factor=1, cells=2, device=None):
    """Address native patch centres in the known 3x3 completed-matrix layout.

    Groups are panel x local spatial cell; they are NOT learned object slots.
    Patches centred in white margins remain outside the branch. Native features
    may already mix across boundaries through the vision encoder's attention.
    """
    h, w = map(int, spatial_shape)
    if min(h, w, factor, cells) < 1:
        raise ValueError('Invalid grid geometry')
    def centers(n):
        start = torch.arange(0, n, factor, device=device).float()
        return (start + (n-start).clamp(max=factor)/2) / n * 520
    yy, xx = torch.meshgrid(centers(h), centers(w), indexing='ij')
    row, col = ((yy-10)/170).floor().long(), ((xx-10)/170).floor().long()
    ly, lx = yy-10-row*170, xx-10-col*170
    valid = (row>=0)&(row<3)&(col>=0)&(col<3)&(ly>=0)&(ly<160)&(lx>=0)&(lx<160)
    ids = ((row*3+col)*cells*cells + (ly*cells/160).floor().long()*cells + (lx*cells/160).floor().long())
    return torch.where(valid, ids, -1).flatten()


def group_pool(x, addresses, count):
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != addresses.numel():
        raise ValueError('Feature grid and group addresses disagree')
    weights = (torch.arange(count, device=x.device)[:,None] == addresses[None,:]).float()
    denom = weights.sum(-1, keepdim=True)
    if bool((denom==0).any()):
        raise ValueError('Native grid too small to populate every panel/cell group')
    return (weights/denom).to(x.dtype) @ x


class NormalizedFusion(nn.Module):
    """(1-alpha)*native + alpha*relation, after token-wise RMS matching.

    Nonzero random projection + initial alpha=.1 lets all branch layers receive
    gradients immediately. Alpha=0 is an explicit bypass check, not initialization.
    """
    def __init__(self, width, output_dim, initial_share=.1):
        super().__init__()
        if not 0 < initial_share < 1:
            raise ValueError('Initial branch share must be strictly between 0 and 1')
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim, bias=False))
        self.logit = nn.Parameter(torch.tensor(math.log(initial_share/(1-initial_share))))
        self.force_share = None
        self.stats = {}

    def forward(self, native, relation):
        if self.force_share == 0:
            return native
        raw = self.projection(relation.float())
        ref = native.detach().float().square().mean(-1,keepdim=True).sqrt()
        rms = raw.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        matched = raw * (ref/rms)
        alpha = self.logit.sigmoid() if self.force_share is None else raw.new_tensor(self.force_share)
        if self.force_share is not None and not 0 <= self.force_share <= 1:
            raise ValueError('Fusion share outside [0,1]')
        result = (1-alpha)*native.float()+alpha*matched
        with torch.no_grad():
            scale = ref.square().mean().sqrt().clamp_min(1e-12)
            self.stats = dict(branch_share=float(alpha.detach()),
                native_rms=float(scale), raw_branch_rms=float(raw.detach().square().mean().sqrt()),
                normalized_branch_to_native_rms=float(matched.detach().square().mean().sqrt()/scale),
                correction_to_native_rms=float((result.detach()-native.float()).square().mean().sqrt()/scale))
        return result.to(native.dtype)


class StagedAbstractor(nn.Module):
    def __init__(self, design, vision_dim=1152, text_dim=2048, width=256, heads=8,
                 depth=6, cells=2, text_groups=16, initial_share=.1, layer_index=2):
        super().__init__()
        if design not in DESIGNS or width % heads or min(depth,cells,text_groups)<1:
            raise ValueError('Invalid design or module dimensions')
        self.spec = dict(design=design, vision_dim=vision_dim, text_dim=text_dim,
                        width=width, heads=heads, depth=depth, cells=cells,
                        text_groups=text_groups, initial_share=initial_share, layer_index=layer_index)
        self.design = design
        dimension = vision_dim if design=='early' else text_dim
        self.group_count = 9*cells*cells
        self.content = nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,width))
        # Addresses supply symbolic values. Content supplies relational Q/K.
        self.visual_symbols = nn.Parameter(torch.randn(1,self.group_count,width)*.02)
        self.layers = nn.ModuleList([AbstractorLayer(width,heads) for _ in range(depth)])
        if design in ('joint','dual'):
            self.text_content = nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,width))
            self.text_symbols = nn.Parameter(torch.randn(1,text_groups,width)*.02)
        if design=='dual':
            self.text_layers = nn.ModuleList([AbstractorLayer(width,heads) for _ in range(depth)])
            self.cross_read = Attention(width,heads)
            self.combine = nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU())
        if design.startswith('memory_'):
            self.query = nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,width))
            self.reader = Attention(width,heads)
        self.fusion = NormalizedFusion(width,dimension,initial_share)

    def abstract(self, visual, addresses, text=None):
        c = self.content(group_pool(visual.float(),addresses,self.group_count))
        states = self.visual_symbols
        nt = 0
        if self.design in ('joint','dual'):
            if text is None or not text.shape[1]:
                raise ValueError('Instruction states are required')
            t = self.text_content(ordered_pool(text.float(),self.spec['text_groups']))
            if self.design=='joint':
                c = torch.cat([c,t],dim=1)
                states = torch.cat([states,self.text_symbols],dim=1)
                nt = t.shape[1]
        for layer in self.layers:
            states = layer(c,states)
        if self.design=='dual':
            ts = self.text_symbols
            for layer in self.text_layers:
                ts = layer(t,ts)
            states = self.combine(torch.cat([states,self.cross_read(c,t,ts)],dim=-1))
        return states[:,:self.group_count], states[:,self.group_count:] if nt else None

    def visual_write(self, native, states, addresses):
        valid = addresses>=0
        result = native.clone()
        if bool(valid.any()):
            result[:,valid] = self.fusion(native[:,valid],states[:,addresses[valid]])
        return result

    def text_write(self, native, states):
        indices = torch.arange(native.shape[1],device=native.device)*states.shape[1]//native.shape[1]
        return self.fusion(native,states[:,indices])

    def memory_read(self, hidden, native, memory):
        return self.fusion(native,self.reader(self.query(hidden.float()),memory,memory))


class Controller:
    def __init__(self,base,processor,module):
        self.base,self.module=base,module
        self.enabled=True
        self.context=None
        self.original_features=None
        self.handles=[]
        self.marker=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
        self.image_id=base.config.image_token_id
        self.factor=int(base.config.downsample_factor)
        index=module.spec['layer_index']
        if base.config.text_config.layer_types[index]!='full_attention':
            raise ValueError('Selected layer must be native full attention')
        self.handles.append(base.register_forward_pre_hook(self.before,with_kwargs=True))
        design=module.design
        if design=='early':
            self.handles.append(base.model.vision_tower.register_forward_hook(self.vision,with_kwargs=True))
        if design in ('late','memory_projected'):
            self.original_features=base.model.get_image_features
            def features(model,*args,**kwargs):
                result=self.original_features(*args,**kwargs)
                if self.enabled:
                    tiles=result.pooler_output
                    if len(tiles)!=1:
                        raise ValueError('Case A expects one whole-image tile')
                    x=tiles[0].unsqueeze(0)
                    states,_=self.module.abstract(x,self.context['projected'])
                    if design=='memory_projected':
                        self.context['memory']=states
                    else:
                        result.pooler_output=[self.module.visual_write(x,states,self.context['projected'])[0]]
                return result
            base.model.get_image_features=MethodType(features,base.model)
        layer=base.model.language_model.layers[index]
        if design in ('joint','dual'):
            self.handles.append(layer.register_forward_hook(self.language))
        if design in ('rca_plugin','memory_projected','memory_contextual'):
            self.handles.append(layer.self_attn.register_forward_hook(self.attention,with_kwargs=True))

    def before(self,base,args,kwargs):
        self.context=None
        if not self.enabled:
            return
        ids=kwargs.get('input_ids')
        shapes=kwargs.get('spatial_shapes')
        mask=kwargs.get('pixel_attention_mask')
        if (ids is None or ids.ndim!=2 or ids.shape[0]!=1 or shapes is None
                or tuple(shapes.shape)!=(1,2) or mask is None):
            raise ValueError('Unpadded single-candidate whole-image prefill required')
        if kwargs.get('past_key_values') is not None or kwargs.get('use_cache',True):
            raise ValueError('Case A supports candidate scoring with use_cache=False only')
        if kwargs.get('labels') is not None or prefix_end(ids[0],self.marker)!=ids.shape[1]:
            raise ValueError('No supervised suffix/labels allowed in candidate input')
        if kwargs.get('attention_mask') is not None and not bool(kwargs['attention_mask'].all()):
            raise ValueError('Padded text is unsupported')
        h,w=map(int,shapes[0].tolist())
        expected=torch.arange(mask.shape[1],device=mask.device)<h*w
        if mask.shape[0]!=1 or not torch.equal(mask[0].bool(),expected):
            raise ValueError('Packed vision mask disagrees with native spatial shape')
        pos=(ids[0]==self.image_id).nonzero().flatten()
        projected=panel_groups((h,w),self.factor,self.module.spec['cells'],ids.device)
        if pos.numel()!=projected.numel() or not pos.numel():
            raise ValueError('Visual token positions disagree with projected patch geometry')
        text=torch.arange(int(pos[-1])+1,ids.shape[1],device=ids.device)
        if text.numel()==0:
            raise ValueError('Expected instruction/assistant header after image')
        self.context=dict(positions=pos,text=text,projected=projected,
                          patches=panel_groups((h,w),1,self.module.spec['cells'],ids.device),memory=None)

    def vision(self,model,args,kwargs,result):
        if self.enabled:
            valid=kwargs['pixel_attention_mask'].bool()
            x=result.last_hidden_state
            selected=x[valid].unsqueeze(0)
            state,_=self.module.abstract(selected,self.context['patches'])
            changed=x.clone()
            changed[valid]=self.module.visual_write(selected,state,self.context['patches'])[0]
            result.last_hidden_state=changed
        return result

    def language(self,model,args,result):
        if not self.enabled:
            return result
        if not isinstance(result,torch.Tensor):
            raise ValueError('Expected native LFM block tensor output')
        ctx=self.context
        visual=result[:,ctx['positions']]
        text=result[:,ctx['text']]
        vs,ts=self.module.abstract(visual,ctx['projected'],text)
        changed=result.clone()
        changed[:,ctx['positions']]=self.module.visual_write(visual,vs,ctx['projected'])
        if self.module.design=='joint':
            changed[:,ctx['text']]=self.module.text_write(text,ts)
        return changed

    def attention(self,model,args,kwargs,result):
        if not self.enabled:
            return result
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        if not isinstance(result,tuple) or hidden is None:
            raise ValueError('Expected native LFM self-attention tuple')
        ctx=self.context
        memory=ctx['memory']
        if self.module.design!='memory_projected':
            memory,_=self.module.abstract(hidden[:,ctx['positions']],ctx['projected'])
        if memory is None:
            raise ValueError('Memory missing for current candidate')
        native=result[0]
        changed=native.clone()
        if self.module.design=='rca_plugin':
            changed[:,ctx['positions']]=self.module.visual_write(native[:,ctx['positions']],memory,ctx['projected'])
        else:
            # Only text AFTER this candidate image can read its memory.
            changed[:,ctx['text']]=self.module.memory_read(hidden[:,ctx['text']],native[:,ctx['text']],memory)
        return (changed,)+result[1:]

    def set_enabled(self,enabled):
        self.enabled=bool(enabled)
        self.context=None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[]
        if self.original_features is not None:
            self.base.model.get_image_features=self.original_features
        self.context=None
        if getattr(self.base,'staged_abstractor',None) is self.module:
            delattr(self.base,'staged_abstractor')
        if getattr(self.base,'_staged_controller',None) is self:
            delattr(self.base,'_staged_controller')


def install(base,processor,design,**options):
    if getattr(base.config,'model_type',None)!='lfm2_vl':
        raise ValueError('Case A currently supports the pinned native LFM2.5 only')
    if any(hasattr(base,n) for n in ('staged_abstractor','relation_probe','abstractor_plugin')):
        raise ValueError('Remove the previous extension before installing Case A')
    if any(p.requires_grad for p in base.parameters()):
        raise ValueError('Freeze all original VLM parameters before installing Case A')
    module=StagedAbstractor(design,vision_dim=base.config.vision_config.hidden_size,
        text_dim=base.config.text_config.hidden_size,**options)
    module.to(device=next(base.parameters()).device,dtype=torch.float32)
    base.add_module('staged_abstractor',module)
    try:
        controller=Controller(base,processor,module)
    except Exception:
        delattr(base,'staged_abstractor')
        raise
    base._staged_controller=controller
    return controller
