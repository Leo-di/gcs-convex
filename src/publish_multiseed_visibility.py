"""Audit a completed paired experiment and publish all three seed replays."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--web', type=Path, required=True)
    args = parser.parse_args()
    assert json.loads((args.source / 'run_status.json').read_text())['status'] == 'completed'
    seeds = json.loads((args.source / 'protocol.json').read_text())['seeds']
    for seed in seeds:
        subprocess.run([sys.executable, '-m', 'src.audit_visibility_clique',
                        '--source', str(args.source / f'seed_{seed}')], check=True)
    subprocess.run([sys.executable, '-m', 'src.summarize_multiseed_visibility',
                    '--source', str(args.source)], check=True)
    aggregate = json.loads((args.source / 'aggregate.json').read_text())
    for seed in seeds:
        subprocess.run([sys.executable, '-m', 'src.report_visibility_clique',
                        '--source', str(args.source / f'seed_{seed}'),
                        '--web', str(args.web / f'seed_{seed}'),
                        '--template', 'src/multiseed_replay_redirect.html',
                        '--query-index', str(aggregate['selected_queries'][str(seed)])], check=True)
    shutil.copy2(args.source / 'aggregate.json', args.web / 'aggregate.json')
    shutil.copy2(args.source / 'protocol.json', args.web / 'study_protocol.json')
    shutil.copy2('src/multiseed_visibility_view.html', args.web / 'index.html')
    print(f'Published audited three-seed experiment to {args.web}', flush=True)


if __name__ == '__main__':
    main()
