#!/usr/bin/env python3
"""Maintain the public package manifest; never generates data or runs science."""
import argparse
import json
from pathlib import Path
from check_repository import ROOT, digest, repository_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Explicitly regenerate FILE_MANIFEST.json after reviewed edits')
    args = parser.parse_args()
    result = {'schema_version': 1, 'files': {name: digest(path) for name, path in repository_files(ROOT)}}
    if args.write:
        path = ROOT / 'FILE_MANIFEST.json'
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        print(f'Wrote {len(result["files"])} file hashes to {path.name}. Run check_repository.py next.')
    else:
        print(json.dumps({'files': len(result['files']), 'written': False, 'requires': '--write'}, indent=2))


if __name__ == '__main__':
    main()
