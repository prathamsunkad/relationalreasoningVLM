import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools import launch_pretrained_connectors as launch

class LauncherTests(unittest.TestCase):
 def test_never_submits_before_pass_and_no_duplicate_cache(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);cache=root/'cache';cfg=root/'plan.json';data=root/'data.jsonl';data.write_text('{}\n')
   plan=dict(max_epochs=1,train_questions=60000,validation_questions=1000,pilot_job_id='1',
    train_manifest=str(data),val_manifest=str(data),train_manifest_sha256=launch.sha(data),val_manifest_sha256=launch.sha(data),
    source_commit='abc',cache_output=str(cache),python='/python',hf_home='/hf',runtime='/runtime')
   cfg.write_text(json.dumps(plan))
   with patch.object(launch,'pilot_passed',return_value=(False,'pending')),patch.object(launch.subprocess,'run') as submit,contextlib.redirect_stdout(io.StringIO()):
    launch.advance(cfg,root,root);submit.assert_not_called()
   with patch.object(launch,'pilot_passed',return_value=(True,'passed')),patch.object(launch,'estimates',return_value={'cache':60}),patch.object(launch.subprocess,'run',return_value=SimpleNamespace(stdout='12345\n')) as submit,contextlib.redirect_stdout(io.StringIO()):
    launch.advance(cfg,root,root);self.assertEqual(submit.call_count,1)
    with patch.object(launch,'job_state',return_value='RUNNING'):
     launch.advance(cfg,root,root);self.assertEqual(submit.call_count,1)
    cache.mkdir();(cache/'failure.json').write_text('{}')
    with patch.object(launch,'job_state',return_value='FAILED'),self.assertRaises(RuntimeError):launch.advance(cfg,root,root)
    self.assertEqual(submit.call_count,1)
   self.assertEqual(len(json.loads((root/'jobs.json').read_text())['jobs']),1)
 def test_per_job_cap(self):
  self.assertEqual(launch.minutes_for(200000),960)
  self.assertEqual(launch.minutes_for(100),60)

if __name__=='__main__':unittest.main()
