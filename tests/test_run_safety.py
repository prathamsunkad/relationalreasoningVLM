"""Guard against silent recipe/data drift without loading pretrained models."""
import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from pgmvlm.config import parse_configured, read_recipe
from pgmvlm.datasets import ManifestStream

class RunSafetyTests(unittest.TestCase):
    def test_recipe_inheritance_and_cli_override(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'base.json').write_text(json.dumps(dict(epochs=3, out_dir='run', flag=True)))
            (root/'child.json').write_text(json.dumps(dict(extends='base.json', epochs=4)))
            parser=argparse.ArgumentParser()
            parser.add_argument('--epochs',type=int)
            parser.add_argument('--out_dir',required=True)
            parser.add_argument('--flag',action='store_true')
            with patch.object(sys,'argv',['test','--config',str(root/'child.json'),'--epochs','5']):
                args=parse_configured(parser)
            self.assertEqual(args.epochs,5)
            self.assertEqual(args.out_dir,'run')
            self.assertTrue(args.flag)
            (root/'base.json').write_text(json.dumps(dict(extends='child.json')))
            with self.assertRaisesRegex(ValueError,'cycle'):read_recipe(root/'child.json')

    def test_clevr_manifest_vocabularies_and_image_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            Image.new('RGB',(32,32),'red').save(root/'image.png')
            digest=hashlib.sha256((root/'image.png').read_bytes()).hexdigest()
            rows=[dict(id=str(n),images=['image.png'],choices=list(choices),answer=choices[-1],
                       prompt='Choose an option.',metadata=dict(contact_sheet_sha256=digest))
                  for n,choices in enumerate(('AB','ABCD'))]
            manifest=root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            stream=ManifestStream(manifest,['0','1'])
            self.assertEqual(stream[0]['target'],1)
            self.assertEqual(stream[1]['target'],3)
            (root/'image.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):stream[0]
            rows[0]['images']=['../outside.png']
            manifest.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'escapes'):ManifestStream(manifest,['0'])[0]

if __name__=='__main__':unittest.main()
