"""Build a finite, checked pickup/action/attached-region library, not select it."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from .animate_delivery_demo import plan_delivery_candidate
from .build_scene import build_scene
from .iris_region_generator import cover_to_dict
from .semantic_gcs_experiment import guards
from .audit_replay import audit_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/results/semantic_candidates/library'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('outputs/results/full_dynamics/reference/scene_config_snapshot.json').read_text())
    (args.output/'config.json').write_text(json.dumps(config))
    report={'scope':'Finite grid at fixed yaw; all candidates evaluated, no first-success selection. Full preliminary plans seed IRIS/action library, not final routes.', 'attempts':[]}
    for x in np.arange(1.10,1.301,.05):
        name=f'x{round(x*100):03d}';q=np.array([round(float(x),2),-.1,0.])
        attempt={'name':name,'base_q':q.tolist()};began=time.perf_counter()
        print('PREPARE '+name,flush=True)
        scene=build_scene(config=copy.deepcopy(config),visualize=False)
        try:
            data=plan_delivery_candidate(scene,q,attempt)
            dense=audit_replay(data['trace'],config,10)
            if not dense['satisfied']:raise RuntimeError('Dense action/reference audit failed: '+str(dense))
            gate=guards(data['trace'],config)
        except RuntimeError as e:
            attempt.update(status='rejected',reason=str(e))
        else:
            directory=args.output/name;directory.mkdir(exist_ok=True)
            for file,value in [('delivery_trace.json',data['trace']),('cover.json',cover_to_dict(data['carrying_iris_cover'])),('guards.json',gate),('dense.json',dense),('scene_config_snapshot.json',config)]:
                (directory/file).write_text(json.dumps(value))
            attempt.update(status='accepted',regions=len(data['carrying_iris_cover'].regions))
        attempt['preparation_seconds']=time.perf_counter()-began
        report['attempts'].append(attempt)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(attempt),flush=True)
    assert sum(a['status']=='accepted' for a in report['attempts'])>=2, 'Need multiple valid candidates before claiming selection'


if __name__=='__main__':main()
