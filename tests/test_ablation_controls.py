"""Image-byte parity and bypass restores native behavior after nonzero modifications."""
import argparse
import unittest
import numpy as np
import torch
from PIL import Image
from tiny_lfm import tiny_lfm
from pgmvlm.panels import split_composite, messages
from pgmvlm.manifests import render_pgm
from pgmvlm.extensions import attach_extension, extension_module, set_extension_enabled
from pgmvlm.protocols import encode_answer
from test_shared_extensions import options
from pgmvlm.evalcore import summarize


class AblationControls(unittest.TestCase):
    def test_separate_panels_are_lossless_ordered_and_same_supervised_prefix(self):
        panels=np.stack([np.full((160,160),i*13,np.uint8) for i in range(16)])
        composite=render_pgm(panels)
        recovered=split_composite(composite)
        self.assertEqual(len(recovered),16)
        for i,p in enumerate(recovered):self.assertTrue((np.array(p)==i*13).all())
        content=messages(composite,'Choose A-H','separate')[0]['content']
        self.assertEqual(content[16]['text'],'Answer option A.')
        self.assertEqual(content[30]['text'],'Answer option H.')
        seen=[]
        class Processor:
            tokenizer=argparse.Namespace(decode=lambda ids,**kw:'A')
            def apply_chat_template(self, turns, add_generation_prompt, **kw):
                seen.append(turns[0])
                ids=[1,2] if add_generation_prompt else [1,2,3,4]
                return {'input_ids':torch.tensor([ids]),'pixel_values':torch.ones(16,3,160,160)}
        encode_answer(Processor(),composite,'Choose A-H','A','cpu',torch.float32,'separate')
        self.assertEqual(len([x for x in seen[0]['content'] if x['type']=='image']),16)
        self.assertEqual(seen[0]['content'],seen[1]['content'])
        with self.assertRaisesRegex(ValueError,'dimensions'):split_composite(Image.new('RGB',(10,10)))

    def test_all_seven_bypasses_restore_native_logits_and_cached_generation(self):
        torch.set_num_threads(1)
        for kind in ('early','late','joint','memory','dual','rca_plugin','rca_memory'):
            with self.subTest(kind=kind):
                model,proc=tiny_lfm();model.requires_grad_(False)
                image=Image.new('RGB',(32,32),'gray')
                prefix,full,start=encode_answer(proc,image,'Choose','A','cpu',torch.float32)
                with torch.no_grad():
                    expected=model(**full,use_cache=False).logits
                    expected_gen=model.generate(**prefix,max_new_tokens=3,do_sample=False,use_cache=True)
                args=options(extension=kind)
                ctl=attach_extension(model,proc,args)
                with torch.no_grad():
                    for p in extension_module(model).parameters():p.add_(.01*torch.randn_like(p))
                model.eval()
                set_extension_enabled(model,ctl,True)
                with torch.no_grad():changed=model(**full,use_cache=False).logits
                self.assertFalse(torch.equal(changed,expected))
                set_extension_enabled(model,ctl,False)
                with torch.no_grad():
                    actual=model(**full,use_cache=False).logits
                    generated=model.generate(**prefix,max_new_tokens=3,do_sample=False,use_cache=True)
                self.assertTrue(torch.equal(actual,expected))
                self.assertTrue(torch.equal(generated,expected_gen))

    def test_mixed_ablation_results_rejected(self):
        row=dict(protocol='answer_letter',choices=list('ABCDEFGH'),target=0,pred=0,correct=True,loss=1.,answer_loss=1.)
        with self.assertRaisesRegex(ValueError,'ablation'):
            summarize([dict(row,extension_enabled=True),dict(row,extension_enabled=False)])
        with self.assertRaisesRegex(ValueError,'ablation'):
            summarize([dict(row,pgm_presentation='separate'),dict(row,pgm_presentation='composite')])

if __name__=='__main__':unittest.main()
