"""Print inherited recipe settings without loading models or starting a job."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pgmvlm.config import read_recipe
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe')
    args = parser.parse_args()
    print(json.dumps(read_recipe(args.recipe), indent=2, sort_keys=True))
