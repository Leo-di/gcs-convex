"""Publish the completed endpoint-attachment pilot, including failed conditions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--web',type=Path,required=True)
    parser.add_argument('--template',type=Path,default=Path('src/online_iris_view.html'))
    args=parser.parse_args()
    assert json.loads((args.source/'status.json').read_text())['stage']=='completed'
    summary=json.loads((args.source/'summary.json').read_text())
    for query in summary['protocol']['queries']:
        folder=f'case_{query:03}'
        if not any(r['query']==query and r['online']['accepted'] for r in summary['rows']):
            target=args.web/folder;target.mkdir(parents=True,exist_ok=True)
            for f in (args.source/folder).glob('*.json'):shutil.copy2(f,target/f.name)
            data=dict(display_query_index=query,recordings=dict(uniform=[],vcc=[]),
                      summary=json.loads((args.source/folder/'summary.json').read_text()))
            (target/'data.json').write_text(json.dumps(data))
            (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><p>本查询两组均无通过审核的轨迹。</p><a href="../">返回全部结果</a>')
            continue
        subprocess.run([sys.executable,'-m','src.report_visibility_clique','--source',str(args.source/folder),
            '--web',str(args.web/folder),'--template','src/multiseed_replay_redirect.html'],check=True)
        data_path=args.web/folder/'data.json'
        data=json.loads(data_path.read_text());data['display_query_index']=query
        data_path.write_text(json.dumps(data))
    for name in ['summary.json','protocol.json']:
        shutil.copy2(args.source/name,args.web/name)
    if (args.source/'analysis.json').exists():
        for name in ['analysis.json','online_region_rejections.json','trajectory_rejection_witnesses.json']:shutil.copy2(args.source/name,args.web/name)
    shutil.copy2(args.template,args.web/'index.html')
    print('Published',args.web,flush=True)

if __name__=='__main__':main()
