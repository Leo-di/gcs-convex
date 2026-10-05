"""Finalize complete paired statistics and link them to the full-task webpage."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .analyze_full_task_evaluation import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/results/full_task_evaluation_40_v1'))
    p.add_argument('--web',type=Path,default=Path('outputs/results/grasp_fix_web/planar/full_task_40'))
    args=p.parse_args()
    if not (args.source/'complete.json').exists():raise RuntimeError('Batch is not complete')
    analysis=analyze(args.source)
    subprocess.run([sys.executable,'-m','src.publish_full_task_development','--source',str(args.source),
                    '--web',str(args.web),'--render'],check=True)
    f=args.web/'summary.json';data=json.loads(f.read_text());data['analysis']=analysis
    f.write_text(json.dumps(data));shutil.copy2(args.source/'analysis.json',args.web/'analysis.json')


if __name__=='__main__':main()
