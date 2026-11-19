"""Export common initial tensors for the depth/readout and insertion-point controls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors.torch import save_file
from pgmvlm.extensions import extension_spec
from pgmvlm.extensions.probe_diagnostics import make_arms, tensor_digest
from pgmvlm.checkpoints import atomic_json


def export(output, seed=20261118, input_dim=2048, width=256, symbols=128,
           model_id='lfm25', revision=None, locations=(2,8,12)):
    output=Path(output)
    if output.exists():
        raise ValueError('Destination exists; initializations are never overwritten')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=output.with_name('.'+output.name+'-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        arms=make_arms(seed,input_dim,width,symbols)
        specs={name:dict(readout='llm' if arm.is_llm else 'classifier_only' if name=='classifier_only' else 'classifier',
                         depth=6 if name.endswith('6') else 2, insertion_layer=None,
                         weights=arm.probe.state_dict() if arm.is_llm else arm.state_dict())
               for name,arm in arms.items()}
        for layer in locations:
            specs[f'location_{layer}']=dict(specs['llm_d2'],insertion_layer=layer)
        receipt=dict(seed=seed,model_id=model_id,revision=revision,input_dim=input_dim,width=width,symbols=symbols,arms={})
        for name,spec in specs.items():
            folder=stage/name;folder.mkdir()
            args=SimpleNamespace(extension='late',extension_width=width,extension_depth=spec['depth'],
                extension_heads=4,extension_symbols=symbols,correction_cap=None,correction_scale=1.,
                readout=spec['readout'],insertion_layer=spec['insertion_layer'])
            save_file({k:v.detach().contiguous() for k,v in spec['weights'].items()},str(folder/'extension.safetensors'))
            atomic_json(folder/'extension.json',extension_spec(args))
            atomic_json(folder/'protocol.json',dict(model_id=model_id,revision=revision,protocol='answer_letter',answer_loss_scope='answer'))
            receipt['arms'][name]=dict(tensor_sha256=tensor_digest(spec['weights'].items()),extension=extension_spec(args))
        atomic_json(stage/'initialization.json',receipt)
        stage.rename(output)
        return receipt
    finally:
        if stage.exists():shutil.rmtree(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--input_dim',type=int,default=2048)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--symbols',type=int,default=128)
    p.add_argument('--model_id',default='lfm25')
    p.add_argument('--revision',default=None)
    args=p.parse_args()
    result=export(**vars(args))
    print(json.dumps(dict(output=args.output,arms=list(result['arms']),seed=args.seed),indent=2))

if __name__=='__main__':main()
