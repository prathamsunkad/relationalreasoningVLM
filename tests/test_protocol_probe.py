import unittest
import numpy as np
from PIL import Image
import torch
from pgmvlm.protocol_probe import candidate_images, candidate_ce, yes_no_score, paired_order
from pgmvlm.manifests import render_pgm


class ProtocolProbeTests(unittest.TestCase):
    def test_lossless_completed_matrices_and_order(self):
        panels=np.stack([np.full((160,160), i*13, dtype=np.uint8) for i in range(16)])
        images=candidate_images(render_pgm(panels))
        self.assertEqual(len(images),8)
        for k,image in enumerate(images):
            self.assertEqual(image.size,(520,520))
            arr=np.asarray(image)
            for position in range(9):
                y=10+(position//3)*170; x=10+(position%3)*170
                expected=panels[position if position<8 else 8+k]
                np.testing.assert_array_equal(arr[y:y+160,x:x+160,0],expected)
            self.assertTrue((arr[0]==255).all())

    def test_candidate_permutation_preserves_loss_and_gradient(self):
        torch.manual_seed(11)
        head=torch.nn.Linear(7,1)
        x=torch.randn(8,7)
        permutation=[3,7,0,6,2,1,5,4]
        scores=head(x).squeeze(-1)
        loss=candidate_ce(scores,2)
        grads=torch.autograd.grad(loss,tuple(head.parameters()))
        perm_loss=candidate_ce(head(x[permutation]).squeeze(-1),permutation.index(2))
        perm_grads=torch.autograd.grad(perm_loss,tuple(head.parameters()))
        torch.testing.assert_close(loss,perm_loss)
        for a,b in zip(grads,perm_grads): torch.testing.assert_close(a,b)

    def test_fp32_yes_no_scoring_and_input_gradient(self):
        torch.manual_seed(3)
        head=torch.nn.Linear(5,20).to(torch.bfloat16).requires_grad_(False)
        hidden=torch.randn(1,5,requires_grad=True)
        score=yes_no_score(hidden,head,4,7)
        full=torch.nn.functional.linear(hidden,head.weight.float(),head.bias.float())
        torch.testing.assert_close(score,(full[:,4]-full[:,7]).squeeze())
        score.backward()
        torch.testing.assert_close(hidden.grad[0],head.weight[4].float()-head.weight[7].float())
        self.assertIsNone(head.weight.grad)

    def test_candidate_ce_is_joint_not_independent_binary(self):
        scores=torch.tensor([0.,1.,2.,3.,4.,5.,6.,7.],requires_grad=True)
        loss=candidate_ce(scores,6)
        loss.backward()
        expected=scores.detach().softmax(0);expected[6]-=1
        torch.testing.assert_close(scores.grad,expected)
        torch.testing.assert_close(candidate_ce(scores.detach()+100,6),loss.detach())
        with self.assertRaises(ValueError): candidate_ce(torch.zeros(16),0)

    def test_paired_order_always_includes_both(self):
        for update in range(10):
            order=paired_order(['answer_letter','candidate'],update)
            self.assertEqual(sorted(order),['answer_letter','candidate'])
            self.assertEqual(order[0],['answer_letter','candidate'][update%2])


if __name__=='__main__': unittest.main()
