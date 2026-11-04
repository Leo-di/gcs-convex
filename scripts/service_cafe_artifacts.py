"""Resolve explicit pickup-cache provenance without modifying planner outputs."""
import json
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
def resolve_pickup(folder):
 folder=Path(folder).resolve();run=folder.parent.parent;seen=set()
 while (folder/'cache_source.json').exists():
  if folder in seen:raise ValueError('Cyclic pickup-cache provenance')
  seen.add(folder);d=json.loads((folder/'cache_source.json').read_text());p=Path(d['source'])
  folder=(p if p.is_absolute() else PROJECT/p).resolve()
  if not folder.is_relative_to(run):raise ValueError('Pickup-cache reference crosses run boundary')
 if not (folder/'reference.json').exists():raise FileNotFoundError(folder/'reference.json')
 return folder
