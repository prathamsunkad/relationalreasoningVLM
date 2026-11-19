"""Small flat JSON recipes; command-line flags override every recipe value."""
import argparse
import json
from pathlib import Path

def read_recipe(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError("Recipe inheritance cycle")
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Recipe must be a JSON object")
    parent = config.pop("extends", None)
    return {**(read_recipe(path.parent / parent, (*ancestors, path)) if parent else {}), **config}



def parse_configured(parser):
    parser.add_argument('--config', help='JSON recipe; CLI values take precedence')
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config')
    known, _ = preliminary.parse_known_args()
    if known.config:
        config = read_recipe(known.config)
        actions = {a.dest: a for a in parser._actions}
        if not isinstance(config, dict) or set(config) - set(actions):
            parser.error('Recipe contains unknown settings')
        for key, value in config.items():
            action = actions[key]
            if action.choices and value not in action.choices:
                parser.error(f'Invalid recipe value for {key}: {value}')
            if action.type and value is not None:
                try:
                    config[key] = action.type(value)
                except (ValueError, TypeError):
                    parser.error(f'Invalid recipe type for {key}')
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) and not isinstance(value, bool):
                parser.error(f'{key} must be a JSON boolean')
            if value is not None:
                action.required = False
        parser.set_defaults(**config)
    return parser.parse_args()
