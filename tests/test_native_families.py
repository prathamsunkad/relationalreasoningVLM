"""Native SmolVLM and Qwen2.5-VL language-only LoRA/answer-loss smoke checks."""
import tempfile
import unittest
import torch
from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from pgmvlm.models import add_lora, load_adapter
from pgmvlm.protocols import answer_losses


def make_native(family):
    torch.manual_seed(23)
    if family=='smolvlm2':
        cfg=SmolVLMConfig(vision_config=dict(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                num_attention_heads=4,image_size=16,patch_size=4),
            text_config=dict(model_type='llama',vocab_size=32,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                pad_token_id=0,bos_token_id=1,eos_token_id=2),image_token_id=3,scale_factor=2)
        model=SmolVLMForConditionalGeneration(cfg)
        ids=torch.tensor([[1,3,3,3,3,5,7,2]])
        data=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                  pixel_values=torch.randn(1,1,3,16,16),pixel_attention_mask=torch.ones(1,1,16,16,dtype=torch.bool))
    else:
        cfg=Qwen2_5_VLConfig(vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
                patch_size=2,temporal_patch_size=1,spatial_merge_size=2,out_hidden_size=32,
                window_size=4,fullatt_block_indexes=[0]),
            text_config=dict(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                num_attention_heads=4,num_key_value_heads=2,rope_parameters=dict(rope_type='default',rope_theta=10000.,mrope_section=[1,1,2]),
                pad_token_id=0,bos_token_id=1,eos_token_id=2),
            image_token_id=3,video_token_id=4,vision_start_token_id=5,vision_end_token_id=6)
        model=Qwen2_5_VLForConditionalGeneration(cfg)
        ids=torch.tensor([[1,5,3,6,8,7,2]])
        data=dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.randn(4,12),
                  image_grid_thw=torch.tensor([[1,2,2]]))
    return model,data

class NativeFamilies(unittest.TestCase):
    def test_answer_loss_lora_freeze_and_reload_for_other_families(self):
        torch.set_num_threads(1)
        for family in ('smolvlm2','spatialevo'):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as td:
                model,data=make_native(family)
                frozen={n:p.detach().clone() for n,p in model.named_parameters()}
                model=add_lora(model,family,r=2,alpha=4,dropout=0.)
                self.assertTrue(any('lora_' in n and p.requires_grad for n,p in model.named_parameters()))
                self.assertFalse(any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n))
                model.eval()
                before=answer_losses(model,data,data['input_ids'].shape[-1]-2)[0]
                before.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for n,p in model.named_parameters() if 'lora_' in n))
                torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01).step()
                for n,p in model.get_base_model().named_parameters():
                    original_name=n.replace('.base_layer.','.')
                    if original_name in frozen:
                        self.assertTrue(torch.equal(p,frozen[original_name]),n)
                with torch.no_grad():expected=answer_losses(model,data,data['input_ids'].shape[-1]-2)[0]
                model.save_pretrained(td,save_embedding_layers=False)
                clone,other=make_native(family)
                clone=load_adapter(clone,td);clone.eval()
                with torch.no_grad():actual=answer_losses(clone,other,other['input_ids'].shape[-1]-2)[0]
                self.assertTrue(torch.equal(actual,expected))

if __name__=='__main__':unittest.main()
