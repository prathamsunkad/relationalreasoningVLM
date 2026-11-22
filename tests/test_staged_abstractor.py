import copy
import unittest
import torch
from pgmvlm.extensions.staged_abstractor import (DESIGNS,NormalizedFusion,
                                               StagedAbstractor,panel_groups,group_pool)


class StagedUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_panel_and_local_addresses(self):
        ids=panel_groups((52,52),cells=2)
        self.assertEqual(set(ids.tolist()),set(range(36))|{-1})
        # Centres (15,15), (95,15), (185,15): panel0 cell0, cell1; panel1 cell0.
        self.assertEqual(ids.reshape(52,52)[1,1],0)
        self.assertEqual(ids.reshape(52,52)[1,9],1)
        self.assertEqual(ids.reshape(52,52)[1,18],4)
        self.assertEqual(ids.reshape(52,52)[0,0],-1)
        features=torch.arange(36).float()[ids.clamp_min(0)].view(1,-1,1)
        torch.testing.assert_close(group_pool(features,ids,36),torch.arange(36).float().view(1,36,1))
        with self.assertRaises(ValueError): group_pool(torch.ones(1,4,2),panel_groups((2,2)),36)

    def test_fusion_is_bounded_and_differentiable(self):
        torch.manual_seed(7)
        fusion=NormalizedFusion(16,32)
        native=torch.randn(1,36,32)
        relation=torch.randn(1,36,16,requires_grad=True)
        output=fusion(native,relation)
        self.assertLessEqual(fusion.stats['correction_to_native_rms'],.20001)
        self.assertAlmostEqual(fusion.stats['normalized_branch_to_native_rms'],1.,places=5)
        output.square().mean().backward()
        self.assertGreater(float(relation.grad.abs().sum()),0)
        fusion.force_share=0.
        self.assertTrue(torch.equal(native,fusion(native,relation)))
        fusion.force_share=1.1
        with self.assertRaises(ValueError): fusion(native,relation)

    def test_all_designs_have_deep_gradients_and_roundtrip(self):
        addresses=panel_groups((16,16),2)
        for design in DESIGNS:
            with self.subTest(design=design):
                torch.manual_seed(19)
                m=StagedAbstractor(design,vision_dim=32,text_dim=32,width=16,heads=4,depth=2)
                x=torch.randn(1,len(addresses),32)
                text=torch.randn(1,9,32)
                states,ts=m.abstract(x,addresses,text if design in ('joint','dual') else None)
                if design.startswith('memory_'): out=m.memory_read(text,text,states)
                else:
                    out=m.visual_write(x,states,addresses)
                    if design=='joint': out=torch.cat([out,m.text_write(text,ts)],dim=1)
                out.square().mean().backward()
                self.assertGreater(float(m.content[1].weight.grad.abs().sum()),0)
                self.assertGreater(float(m.layers[0].relation.q.weight.grad.abs().sum()),0)
                self.assertGreater(float(m.layers[-1].relation.q.weight.grad.abs().sum()),0)
                copy_m=StagedAbstractor(**m.spec)
                copy_m.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
                for a,b in zip(m.parameters(),copy_m.parameters()): self.assertTrue(torch.equal(a,b))
                if not design.startswith('memory_'):
                    self.assertTrue(torch.equal(x[:,addresses<0],out[:,:len(addresses)][:,addresses<0]))

    def test_digest_accepts_scalar_gates_and_preserves_existing_hashes(self):
        import hashlib
        from pgmvlm.extensions.probe_diagnostics import tensor_digest
        for dtype in (torch.float32,torch.bfloat16):
            value=torch.tensor(.1,dtype=dtype)
            first=tensor_digest([('gate',value)])
            self.assertEqual(first,tensor_digest([('gate',value.clone())]))
            self.assertNotEqual(first,tensor_digest([('gate',value+1)]))
        matrix=torch.randn(3,5).t()
        old=hashlib.sha256()
        old.update(b'matrix');old.update(matrix.contiguous().view(torch.uint8).numpy().tobytes())
        self.assertEqual(old.hexdigest(),tensor_digest([('matrix',matrix)]))

    def test_schedule(self):
        from tools.staged_abstractor_pilot import lr_at
        values=[lr_at(i,16,3e-4,2,.1) for i in range(16)]
        self.assertAlmostEqual(values[0],1.5e-4)
        self.assertAlmostEqual(values[1],3e-4)
        self.assertAlmostEqual(values[-1],3e-5)
        self.assertTrue(all(a>=b for a,b in zip(values[2:],values[3:])))


if __name__=='__main__': unittest.main()
