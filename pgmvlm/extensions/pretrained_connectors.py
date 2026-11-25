"""Three interfaces to frozen external relations; no Abstractor/LoRA updates.

Native LFM2-VL only. Serial, unpadded, full-prefix forwards (use_cache=False)
are intentional: training and greedy decoding use exactly the same hooks.
Memory is supplied explicitly per question, never constructed from labels.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F

DESIGNS = ('reader', 'kv_memory', 'tokens')


def bound(delta, native, cap):
    """Clip excessive corrections; never amplify tiny/zero corrections."""
    limit = cap * native.detach().float().square().mean(-1, keepdim=True).sqrt()
    rms = delta.float().square().mean(-1, keepdim=True).clamp_min(1e-24).sqrt()
    return (delta.float() * (limit / rms).clamp(max=1)).to(native.dtype)


class Connector(nn.Module):
    def __init__(self, design, hidden, kv_heads, head_dim, width=256, heads=8, cap=.1):
        super().__init__()
        if design not in DESIGNS or width % heads or not 0 < cap <= 1:
            raise ValueError('Invalid connector specification')
        self.spec = dict(design=design, hidden=hidden, kv_heads=kv_heads, head_dim=head_dim,
                         width=width, heads=heads, cap=cap, memory_shape=[8,9,32])
        self.design, self.cap = design, cap
        self.norm = nn.LayerNorm(32)
        self.input = nn.Linear(32, width)
        self.candidate = nn.Embedding(8, width)
        self.panel = nn.Embedding(9, width)
        nn.init.normal_(self.candidate.weight, std=.02)
        nn.init.normal_(self.panel.weight, std=.02)
        if design == 'reader':
            self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width))
            self.reader = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
            self.output = nn.Linear(width, hidden, bias=False)
            nn.init.zeros_(self.output.weight)
        elif design == 'kv_memory':
            self.kv = nn.Linear(width, 2*kv_heads*head_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.output = nn.Linear(width, hidden, bias=False)
            nn.init.normal_(self.output.weight, std=.002)

    def memory(self, bank):
        if tuple(bank.shape) != (1,8,9,32) or bank.requires_grad:
            raise ValueError('Expected one detached 8-candidate x 9-panel x 32-feature bank')
        ids = torch.arange(8,device=bank.device)
        panels = torch.arange(9,device=bank.device)
        x = self.input(self.norm(bank.float()))
        return (x + self.candidate(ids)[None,:,None] + self.panel(panels)[None,None]).reshape(1,72,-1)

    def read(self, hidden, memory):
        q = self.query(hidden.float())
        read,_ = self.reader(q,memory,memory,need_weights=False)
        delta = self.output(read)
        return hidden + bound(delta,hidden,self.cap)


class Controller:
    def __init__(self, base, module, layer_index):
        self.base, self.module = base,module
        self.enabled, self.bank, self.context = True,None,None
        self.handles=[]
        self.language=base.model.language_model
        self.handles.append(base.register_forward_pre_hook(self.prepare,with_kwargs=True))
        if module.design=='reader':
            self.handles.append(self.language.layers[layer_index].register_forward_hook(self.read))
        elif module.design=='kv_memory':
            attn=self.language.layers[layer_index].self_attn
            self.handles.append(attn.register_forward_hook(self.kv,with_kwargs=True))
        else:
            self.handles.append(self.language.register_forward_pre_hook(self.insert,with_kwargs=True))
            self.handles.append(self.language.register_forward_hook(self.strip))

    @contextmanager
    def question(self, bank):
        if self.bank is not None:
            raise RuntimeError('Nested or stale question memory')
        self.bank=bank
        try:
            yield self
        finally:
            self.bank=None; self.context=None

    def prepare(self, model, args, kwargs):
        self.context=None
        if not self.enabled:
            return
        ids=kwargs.get('input_ids')
        if args or ids is None or ids.ndim!=2 or ids.shape[0]!=1:
            raise ValueError('Use keyword inputs, one unpadded question at a time')
        if kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None:
            raise ValueError('Connectors require full-prefix use_cache=False')
        if kwargs.get('labels') is not None or kwargs.get('position_ids') is not None:
            raise ValueError('Compute answer loss externally; native positions only')
        mask=kwargs.get('attention_mask')
        if mask is not None and (mask.shape!=ids.shape or not bool(mask.all())):
            raise ValueError('Padded/nonstandard attention mask unsupported')
        if self.bank is None:
            raise ValueError('Supply memory through controller.question for every question')
        pos=(ids[0]==self.base.config.image_token_id).nonzero().flatten()
        if not len(pos) or int(pos[-1])+1>=ids.shape[1]:
            raise ValueError('Expected image followed by instruction and assistant prefix')
        self.context=dict(start=int(pos[-1])+1, length=ids.shape[1], memory=self.module.memory(self.bank))

    def read(self, model, args, result):
        if not self.enabled: return result
        if not isinstance(result,torch.Tensor): raise ValueError('Native LFM block must return Tensor')
        start=self.context['start']
        return torch.cat((result[:,:start],self.module.read(result[:,start:],self.context['memory'])),dim=1)

    def kv(self, attn, args, kwargs, result):
        if not self.enabled: return result
        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb, repeat_kv
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        if hidden is None or kwargs.get('past_key_values') is not None:
            raise ValueError('Unsupported native attention invocation')
        # Reuse native Q/K/V and normalization, including frozen historical LoRA.
        shape=(*hidden.shape[:-1],-1,attn.head_dim)
        q=attn.q_layernorm(attn.q_proj(hidden).view(shape)).transpose(1,2)
        k=attn.k_layernorm(attn.k_proj(hidden).view(shape)).transpose(1,2)
        v=attn.v_proj(hidden).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
        k=repeat_kv(k,attn.num_key_value_groups);v=repeat_kv(v,attn.num_key_value_groups)
        start=self.context['start'];q=q[:,:,start:]
        mask=kwargs.get('attention_mask')
        if mask is None:
            qi=torch.arange(start,hidden.shape[1],device=hidden.device)[:,None]
            ki=torch.arange(hidden.shape[1],device=hidden.device)[None,:]
            mask=torch.zeros((1,1,q.shape[2],hidden.shape[1]),device=hidden.device,dtype=q.dtype)
            mask=mask.masked_fill(ki>qi,torch.finfo(q.dtype).min)
        else:
            mask=mask[:,:,start:,:k.shape[2]]
            if mask.dtype==torch.bool:
                mask=torch.zeros_like(mask,dtype=q.dtype).masked_fill(~mask,torch.finfo(q.dtype).min)
        s=self.module.spec
        mk,mv=self.module.kv(self.context['memory']).view(1,72,2,s['kv_heads'],s['head_dim']).unbind(2)
        mk=repeat_kv(mk.transpose(1,2).to(q.dtype),attn.num_key_value_groups)
        mv=repeat_kv(mv.transpose(1,2).to(q.dtype),attn.num_key_value_groups)
        # Memory keys live in post-RoPE space: candidate/panel IDs, no fake temporal positions.
        # A joint softmax lets memory compete with native context (not a separate reader).
        native=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=attn.scaling)
        joint_mask=F.pad(mask,(0,72),value=0.)
        joint=F.scaled_dot_product_attention(q,torch.cat((k,mk),2),torch.cat((v,mv),2),
                     attn_mask=joint_mask,dropout_p=0.,scale=attn.scaling)
        delta=(joint-native).transpose(1,2).reshape(1,q.shape[2],-1)
        # Project a difference through the native linear map; subtract bias if any.
        delta=attn.out_proj(delta)
        if getattr(attn.out_proj,'bias',None) is not None: delta=delta-attn.out_proj.bias
        delta=delta*torch.tanh(self.module.gate).to(delta.dtype)
        output=result[0]
        changed=torch.cat((output[:,:start],output[:,start:]+bound(delta,output[:,start:],self.module.cap)),1)
        return (changed,)+result[1:]

    def insert(self, model, args, kwargs):
        if not self.enabled: return
        if args or kwargs.get('inputs_embeds') is None or kwargs.get('past_key_values') is not None:
            raise ValueError('Expected native merged multimodal embeddings without cache')
        x=kwargs['inputs_embeds'];start=self.context['start']
        tokens=self.module.output(self.context['memory']).to(x.dtype)
        native_scale=x[:,:start].detach().float().square().mean().sqrt()
        rms=tokens.float().square().mean(-1,keepdim=True).clamp_min(1e-24).sqrt()
        tokens=tokens*(native_scale/rms).clamp(max=1).to(tokens.dtype)
        kw=dict(kwargs)
        kw['inputs_embeds']=torch.cat((x[:,:start],tokens,x[:,start:]),1)
        kw['input_ids']=None
        kw['attention_mask']=torch.ones(kw['inputs_embeds'].shape[:2],device=x.device,dtype=torch.long)
        kw['position_ids']=None;kw['cache_position']=None
        return args,kw

    def strip(self, model, args, result):
        if not self.enabled: return result
        start=self.context['start']
        result.last_hidden_state=torch.cat((result.last_hidden_state[:,:start],result.last_hidden_state[:,start+72:]),1)
        if result.hidden_states is not None:
            result.hidden_states=tuple(torch.cat((h[:,:start],h[:,start+72:]),1) for h in result.hidden_states)
        return result

    def set_enabled(self,enabled):
        self.enabled=bool(enabled);self.context=None

    def remove(self):
        for h in self.handles: h.remove()
        self.handles=[];self.bank=None;self.context=None
        delattr(self.base,'pretrained_connector')


def install(base,design,layer_index=8,**options):
    if base.config.model_type!='lfm2_vl' or any(p.requires_grad for p in base.parameters()):
        raise ValueError('Requires completely frozen native LFM2-VL including LoRA')
    if any(hasattr(base,n) for n in ('pretrained_connector','staged_abstractor','relation_probe','abstractor_plugin')):
        raise ValueError('Remove previous extension before installing connector')
    config=base.config.text_config
    if design!='tokens' and config.layer_types[layer_index]!='full_attention':
        raise ValueError('Reader/KV comparison must use the selected full-attention layer')
    module=Connector(design,config.hidden_size,config.num_key_value_heads,
                     getattr(config,'head_dim',config.hidden_size//config.num_attention_heads),**options)
    module.to(device=next(base.parameters()).device,dtype=torch.float32)
    base.add_module('pretrained_connector',module)
    ctl=Controller(base,module,layer_index)
    module.spec['layer_index']=layer_index if design!='tokens' else None
    return ctl
