"""Read-only audit of one or more training run directories."""

import argparse
import json

from wild_drone.checkpoint_audit import audit_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps({'directory': directory, **audit_checkpoints(directory)}, indent=2))


if __name__ == '__main__':
    main()
