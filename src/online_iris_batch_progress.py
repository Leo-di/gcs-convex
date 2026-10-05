"""Publish live batch progress without treating pending queries as failures."""
import argparse
import json
from pathlib import Path
import time


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return default


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--web',type=Path,required=True)
    args=p.parse_args();args.web.mkdir(parents=True,exist_ok=True)
    while True:
        protocol=read(args.source/'protocol.json',{});status=read(args.source/'status.json',{})
        infos=[read(f) for f in args.source.glob('case_*/*_attachment.json')];infos=[i for i in infos if i is not None]
        summary=read(args.source/'summary.json',{});rows=summary.get('rows',[])
        value=dict(stage=status.get('stage','starting'),queries=len(protocol.get('queries',[])),
            elapsed_seconds=time.time()-(args.source/'protocol.json').stat().st_mtime,
            completed_attachment_conditions=len(infos),iris_calls_recorded=sum(i['attempts'] for i in infos),
            evaluated_conditions=len(rows),current=status,
            methods={m:dict(evaluated=sum(r['method']==m for r in rows),offline_successes=sum(r['method']==m and r['offline']['accepted'] for r in rows),online_successes=sum(r['method']==m and r['online']['accepted'] for r in rows)) for m in ['uniform','vcc']})
        tmp=args.web/'progress.tmp';tmp.write_text(json.dumps(value));tmp.replace(args.web/'progress.json')
        if status.get('stage')=='completed':break
        time.sleep(10)

if __name__=='__main__':main()
