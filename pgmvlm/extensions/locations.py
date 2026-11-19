"""Visual-only insertion after a decoder block, sharing the original abstractor."""
import torch
from .relational import RelationalProbe


class LocationController:
    def __init__(self, base, index, width, heads, depth, symbols):
        types = base.config.text_config.layer_types
        if not 0 <= index < len(types)-1 or 'full_attention' not in types[index+1:]:
            raise ValueError('Insertion needs a downstream full-attention layer to read the visual correction')
        self.probe = RelationalProbe('late', text_dim=base.config.text_config.hidden_size,
                                     width=width, heads=heads, depth=depth, symbols=symbols)
        self.probe.to(device=next(base.parameters()).device, dtype=torch.float32)
        base.add_module('relation_probe', self.probe)
        self.mask = None
        self.image_id = base.config.image_token_id
        self.handles = [base.register_forward_pre_hook(self.before, with_kwargs=True),
                        base.model.language_model.layers[index].register_forward_hook(self.after)]

    def before(self, module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None or ids.shape[0] != 1:
            raise ValueError('Location probes require one unpadded native input')
        mask = kwargs.get('attention_mask')
        if mask is not None and not bool(mask.all()):
            raise ValueError('Padded inputs are unsupported by location probes')
        self.mask = ids[0] == self.image_id if kwargs.get('pixel_values') is not None else None

    def after(self, module, args, output):
        if self.mask is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        changed[:, self.mask] = self.probe(hidden[:, self.mask])
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
