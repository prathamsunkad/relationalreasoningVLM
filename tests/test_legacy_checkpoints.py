"""Historical containers round-trip to shared adapters without losing tensors."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file, load_file
from pgmvlm.legacy_checkpoints import convert, build_module
from pgmvlm.checkpoints import sha256


def arguments(extension='late'):
    return argparse.Namespace(extension=extension,extension_width=16,extension_depth=2,
        extension_heads=4,extension_symbols=8,correction_cap=None,correction_scale=1.,
        readout='llm',insertion_layer=None,model_id='lfm25',revision=None,
        protocol='answer_letter',answer_loss_scope='suffix',text_dim=32,vision_dim=32)


class LegacyCheckpoints(unittest.TestCase):
    def test_all_original_probes_frozen_or_joint(self):
        for variant in ('early','late','joint','memory','dual'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as td:
                root=Path(td);old=root/'historical';old.mkdir();args=arguments(variant)
                weights=build_module(args).state_dict();save_file(weights,str(old/'probe.safetensors'))
                (old/'checksums.json').write_text(json.dumps({'probe.safetensors':sha256(old/'probe.safetensors')}))
                result=convert(old,root/'converted',args)
                self.assertFalse(result['optimizer_imported'])
                for name,tensor in load_file(str(root/'converted/extension.safetensors')).items():
                    self.assertTrue(torch.equal(tensor,weights[name]))
                self.assertFalse((root/'converted/state.pt').exists())
                with self.assertRaisesRegex(ValueError,'already exists'):convert(old,root/'converted',args)
                (old/'probe.safetensors').write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError,'checksum'):convert(old,root/'bad',args)

    def test_packed_plugin_and_memory_with_lora_and_scale_validation(self):
        for kind in ('rca_plugin','rca_memory'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);old=root/'historical';old.mkdir();args=arguments(kind)
                args.correction_scale=.01
                module=build_module(args)
                weights={'base_model.model.abstractor_plugin.'+k:v for k,v in module.state_dict().items()}
                prefix='base_model.model.model.language_model.layers.2.self_attn.q_proj.'
                weights[prefix+'lora_A.default.weight']=torch.ones(2,32)
                weights[prefix+'lora_B.default.weight']=torch.ones(32,2)
                torch.save(weights,old/'weights.pt')
                manifest={'schema_version':1,'files':{'weights.pt':sha256(old/'weights.pt')},'identity':{
                    'plan':{'plugin_output_scale':.01,'lora':{'r':2,'lora_alpha':4,'lora_dropout':.05}}}}
                (old/'manifest.json').write_text(json.dumps(manifest))
                convert(old,root/'converted',args)
                lora=load_file(str(root/'converted/adapter_model.safetensors'))
                self.assertEqual(len(lora),2)
                self.assertTrue(all('.default.' not in k for k in lora))
                self.assertEqual(json.loads((root/'converted/adapter_config.json').read_text())['r'],2)
                args.correction_scale=1.
                with self.assertRaisesRegex(ValueError,'scale'):convert(old,root/'bad',args)

    def test_diagnostic_arm_and_safe_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=arguments('late');module=build_module(args)
            torch.save({'arm':{'probe.'+k:v for k,v in module.state_dict().items()}},root/'best.pt')
            convert(root/'best.pt',root/'converted',args)
            (root/'best.json').write_text(json.dumps({'directory':'../escape'}))
            with self.assertRaisesRegex(ValueError,'Unsafe'):convert(root,root/'bad',args)


class NativeLegacyReload(unittest.TestCase):
    def test_joint_packed_conversion_preserves_logits_and_trainability(self):
        from tiny_lfm import tiny_lfm
        from pgmvlm.models import add_lora, load_adapter
        from pgmvlm.extensions import attach_extension, extension_module
        from pgmvlm.protocols import encode_answer, answer_losses
        from PIL import Image
        torch.set_num_threads(1)
        def fixture():
            # Include downstream attention so visual corrections reach the answer.
            from transformers import Lfm2VlForConditionalGeneration
            model,proc=tiny_lfm()
            model.config.text_config.layer_types[-1]='full_attention'
            torch.manual_seed(0)
            return Lfm2VlForConditionalGeneration(model.config),proc
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=root/'old';old.mkdir()
            args=arguments('rca_plugin');args.model_id='tiny_lfm25'
            model,proc=fixture()
            model=add_lora(model,'lfm25',2,4,0.,False,False,False)
            attach_extension(model,proc,args)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:p.add_(.01*torch.randn_like(p))
            model.eval()
            _,full,start=encode_answer(proc,Image.new('RGB',(32,32),'gray'),'Choose','A','cpu',torch.float32)
            with torch.no_grad():expected=model(**full,use_cache=False).logits
            weights={n:p.detach().clone() for n,p in model.named_parameters() if 'lora_' in n or 'abstractor_plugin.' in n}
            torch.save(weights,old/'weights.pt')
            model.save_pretrained(root/'config',safe_serialization=True)
            convert(old,root/'converted',args,adapter_config=root/'config/adapter_config.json')
            clone,processor=fixture()
            clone=load_adapter(clone,root/'converted',trainable=True)
            attach_extension(clone,processor,args,root/'converted')
            clone.eval()
            with torch.no_grad():actual=clone(**full,use_cache=False).logits
            self.assertTrue(torch.equal(expected,actual))
            loss,_,_=answer_losses(clone,full,start);loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in extension_module(clone).parameters()))
            self.assertTrue(all(p.grad is None for p in clone.parameters() if not p.requires_grad))

if __name__=='__main__':
    unittest.main()
