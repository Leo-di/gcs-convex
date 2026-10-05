"""Frozen-library ablation: one solve, retries only, or cuts plus retries."""
import json,copy,shutil,sys,subprocess
from pathlib import Path
from .benchmark_semantic_baselines import run,atomic_json,file_hash

def main():
 root=Path('outputs/results/online_repair_ablation_v1')
 if root.exists():raise RuntimeError('Refusing to overwrite experiment')
 root.mkdir();base=json.loads(Path('outputs/results/semantic_multiscene_v2/protocol.json').read_text())
 snapshot=root/'source_snapshot';snapshot.mkdir()
 for p in Path('src').glob('*.py'):shutil.copyfile(p,snapshot/p.name)
 paths=sorted(Path('src').glob('*.py'))+sorted(Path('outputs/results/semantic_multiscene_v2/libraries').rglob('*.json'))
 design={'arms':['none','retry','cuts'],'rounds':3,'budget_seconds':300,'seeds':[0,7,19],
 'scope':'Same frozen v2 libraries; every arm rerun with identical code and budgets. Retry and cuts use same seed sequence seed+101*iteration, at most 4 solves per candidate subset. Cuts only on carrying collisions, no relaxed audit. B1/B2 single-candidate and joint multi-candidate semantics unchanged. No library costs rerun.',
 'contacts':'Cuts arm only: all four scenes, seed 0, all three methods; other seeds planning only.',
 'hashes':{str(p):file_hash(p) for p in paths}}
 atomic_json(root/'design.json',design)
 for mode in design['arms']:
  out=root/mode;out.mkdir();spec=copy.deepcopy(base)
  spec.update(repair_mode=mode,repair_rounds=0 if mode=='none' else 3,input_sha256=design['hashes'])
  atomic_json(out/'protocol.json',spec);run(out,spec)
 subprocess.run([sys.executable,'-m','src.benchmark_semantic_contacts','--root',str(root/'cuts')],check=True)
if __name__=='__main__':main()
