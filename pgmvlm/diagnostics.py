"""Opt-in gradient measurements in the shared trainer; values only, no retained graphs."""
import torch


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


class GradientRecorder:
    def __init__(self, model):
        self.records = {}
        self.handles = []
        self.enabled = False
        for name, module in model.named_modules():
            relevant = (name.endswith(('vision_tower', 'multi_modal_projector', 'relation_probe', 'abstractor_plugin'))
                        or '.language_model.layers.' in name and name.rsplit('.', 1)[-1].isdigit()
                        or any(marker in name for marker in ('.relation.', '.self_attention.', '.reader_')))
            if relevant:
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def capture(module, inputs, output):
            if not self.enabled:
                return
            value = output
            if hasattr(output, 'last_hidden_state'):
                value = output.last_hidden_state
            elif isinstance(output, tuple):
                value = output[0]
            if not isinstance(value, torch.Tensor):
                return
            entry = dict(activation_rms=rms(value), elements=value.numel(), requires_grad=value.requires_grad)
            self.records[name] = entry
            if value.requires_grad:
                def gradient(g):
                    entry.update(gradient_rms=rms(g), gradient_l2=float(g.detach().float().norm()),
                                 gradient_to_activation_rms=rms(g)/max(entry['activation_rms'], 1e-30),
                                 finite=bool(torch.isfinite(g).all()))
                value.register_hook(gradient)
        return capture

    def start(self):
        self.enabled = True
        self.records = {}

    def finish(self, model):
        self.enabled = False
        parameters = {name: dict(parameter_rms=rms(p), gradient_rms=rms(p.grad) if p.grad is not None else None,
                                 gradient_l2=float(p.grad.detach().float().norm()) if p.grad is not None else None)
                      for name, p in model.named_parameters() if p.requires_grad}
        return dict(activations=self.records, parameters=parameters,
                    note='Activation gradients from final microbatch; parameter gradients accumulated over global batch; before clipping')

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
