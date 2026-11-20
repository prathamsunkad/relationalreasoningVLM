"""Offline conversion of historical adapters/probes/plugin snapshots. No base download."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pgmvlm.config import read_recipe
from pgmvlm.extensions import add_extension_args
from pgmvlm.protocols import add_protocol_args
from pgmvlm.legacy_checkpoints import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_extension_args(parser)
    add_protocol_args(parser)
    parser.set_defaults(protocol='answer_letter')
    parser.add_argument('--model_id', default='lfm25')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--config', help='Training recipe; only architecture/protocol fields are used')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pointer', choices=['best','latest'], default='best')
    parser.add_argument('--adapter-config', default=None)
    parser.add_argument('--text_dim', type=int, default=2048)
    parser.add_argument('--vision_dim', type=int, default=1152)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        keys = {action.dest for action in parser._actions} - {'source','output','pointer','adapter_config','config'}
        parser.set_defaults(**{k:v for k,v in read_recipe(preliminary.config).items() if k in keys})
    args = parser.parse_args()
    result = convert(args.source, args.output, args, args.pointer, args.adapter_config)
    print(json.dumps(dict(output=args.output, source=result['resolved_source'],
                          optimizer_imported=False), indent=2))

if __name__ == '__main__':
    main()
