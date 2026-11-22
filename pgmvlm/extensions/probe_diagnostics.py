"""Depth/readout diagnostic kernels imported from the original experiments."""
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .relational import RelationalProbe, ordered_pool
from . import bounded_correction
ARM_NAMES = ("llm_d2", "llm_d6", "classifier_d2", "classifier_d6", "classifier_only")

class PositionClassifier(nn.Module):
    """Shared small readout preserving bin order; no attention or object slots."""
    def __init__(self, width=256, bins=128):
        super().__init__()
        self.compress = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU())
        self.classify = nn.Sequential(nn.Flatten(1), nn.Linear(bins * 32, 128),
                                      nn.GELU(), nn.Linear(128, 8))

    def forward(self, x):
        return self.classify(self.compress(x))

class DiagnosticArm(nn.Module):
    def __init__(self, name, input_dim=2048, width=256, bins=128):
        super().__init__()
        if name not in ARM_NAMES:
            raise ValueError(name)
        self.name = name
        self.is_llm = name.startswith('llm_')
        self.bins = bins
        self.max_correction_ratio = None
        if name == 'classifier_only':
            self.content_in = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        else:
            self.probe = RelationalProbe('late', text_dim=input_dim, width=width,
                                         heads=4, depth=int(name[-1]), symbols=bins)
            if not self.is_llm:
                # Direct symbol classifier never uses the LLM writeback interface.
                self.probe.read.requires_grad_(False)
                self.probe.output.requires_grad_(False)
        if not self.is_llm:
            self.head = PositionClassifier(width, bins)

    def forward(self, x, diagnostics=None):
        if self.name == 'classifier_only':
            representation = ordered_pool(self.content_in(x.float()), self.bins)
        else:
            c, pooled, representation = self.probe.abstract(x)
        if diagnostics is not None:
            watch_tensor(diagnostics, 'abstractor_symbols' if self.name != 'classifier_only'
                         else 'projected_content', representation)
        if not self.is_llm:
            return self.head(representation)
        delta = self.probe.output(self.probe.read(c, pooled, representation)).to(x.dtype)
        raw_delta = delta
        delta = bounded_correction(delta, x, self.max_correction_ratio)
        if diagnostics is not None:
            diagnostics['raw_correction_input_rms_ratio'] = rms(raw_delta) / max(rms(x), 1e-30)
            diagnostics['correction_input_rms_ratio'] = rms(delta) / max(rms(x), 1e-30)
            diagnostics['correction_cap'] = self.max_correction_ratio
            watch_tensor(diagnostics, 'correction', delta)
        return x + delta

def make_arms(seed, input_dim=2048, width=256, bins=128):
    """Share all common initial tensors; extra depth gets its own fresh layers."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        two = DiagnosticArm('llm_d2', input_dim, width, bins)
        torch.manual_seed(seed + 1)
        six = DiagnosticArm('llm_d6', input_dim, width, bins)
        state = six.probe.state_dict()
        state.update(copy.deepcopy(two.probe.state_dict()))
        six.probe.load_state_dict(state)
        arms = {'llm_d2': two, 'llm_d6': six}
        torch.manual_seed(seed + 2)
        head = PositionClassifier(width, bins)
        for depth in (2, 6):
            arm = DiagnosticArm(f'classifier_d{depth}', input_dim, width, bins)
            arm.probe.load_state_dict(arms[f'llm_d{depth}'].probe.state_dict())
            arm.head.load_state_dict(head.state_dict())
            arms[arm.name] = arm
        control = DiagnosticArm('classifier_only', input_dim, width, bins)
        control.content_in.load_state_dict(two.probe.content_in.state_dict())
        control.head.load_state_dict(head.state_dict())
        arms[control.name] = control
    return arms

def tensor_digest(named):
    digest = hashlib.sha256()
    for name, tensor in named:
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()

def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())

def watch_tensor(record, name, tensor, image_mask=None, answer_position=None):
    if not tensor.requires_grad:
        record[name] = {'requires_grad': False}
        return
    scopes = {'all': None}
    if image_mask is not None:
        scopes['visual'] = image_mask
    if answer_position is not None:
        scopes['answer_query'] = slice(answer_position, answer_position + 1)
    entry = {'requires_grad': True}
    for scope, index in scopes.items():
        values = tensor if index is None else tensor[:, index]
        entry[scope] = {'activation_rms': rms(values), 'elements': values.numel()}
    record[name] = entry

    def backward(gradient):
        for scope, index in scopes.items():
            values = gradient if index is None else gradient[:, index]
            entry[scope].update(gradient_rms=rms(values),
                                gradient_l2=float(values.detach().float().norm()),
                                finite=bool(torch.isfinite(values).all()))
    tensor.register_hook(backward)

def llm_logits(model, arm, sample, diagnostics=None):
    """Only prompt embeddings enter the LLM; the answer is never in this input."""
    embeds, mask = sample['embeds'], sample['image_mask']
    corrected = arm(embeds[:, mask], diagnostics)
    # Differentiable scatter through the entire frozen decoder is essential.
    combined = embeds.clone()
    combined[:, mask] = corrected
    if diagnostics is not None:
        watch_tensor(diagnostics, 'llm_input', combined, mask, combined.shape[1] - 1)
    handles = []
    if diagnostics is not None:
        for index, layer in enumerate(model.model.language_model.layers):
            def hook(module, args, output, index=index):
                value = output[0] if isinstance(output, tuple) else output
                watch_tensor(diagnostics, f'llm_layer_{index:02d}', value, mask, value.shape[1] - 1)
            handles.append(layer.register_forward_hook(hook))
    try:
        return model(inputs_embeds=combined, attention_mask=sample['attention_mask'],
                     use_cache=False, logits_to_keep=1).logits[:, -1].float()
    finally:
        for handle in handles:
            handle.remove()

def answer_loss(model, arm, sample, diagnostics=None):
    if arm.is_llm:
        logits = llm_logits(model, arm, sample, diagnostics)
        target = sample['answer_token'].reshape(1)
    else:
        logits = arm(sample['embeds'][:, sample['image_mask']], diagnostics)
        target = sample['answer_class'].reshape(1)
    return F.cross_entropy(logits, target), int(logits.argmax(-1)), logits

def gradient_stats(module):
    result = {}
    for name, p in module.named_parameters():
        if p.requires_grad:
            result[name] = {'parameter_rms': rms(p), 'gradient_present': p.grad is not None,
                            'gradient_rms': rms(p.grad) if p.grad is not None else None,
                            'gradient_l2': float(p.grad.float().norm()) if p.grad is not None else None}
    return result

def native_prompt_sample(model, inputs, labels, start):
    """Exactly reproduce native LFM visual scatter, then discard the answer suffix."""
    with torch.no_grad():
        features = model.model.get_image_features(
            **{key: inputs[key] for key in ('pixel_values', 'spatial_shapes', 'pixel_attention_mask')},
            return_dict=True).pooler_output
        image = torch.cat(features, dim=0)
        embeds = model.get_input_embeddings()(inputs['input_ids'])
        scatter = model.model.get_placeholder_mask(inputs['input_ids'], embeds, image)
        embeds = embeds.masked_scatter(scatter, image.to(embeds.dtype))[:, :start].contiguous()
    mask = inputs['input_ids'][0, :start] == model.config.image_token_id
    if not mask.any() or mask.sum() != image.shape[0]:
        raise ValueError('Native image features and token mask disagree')
    return {'embeds': embeds, 'image_mask': mask,
            'attention_mask': inputs['attention_mask'][:, :start].contiguous(),
            'answer_token': labels[0, start].clone()}
