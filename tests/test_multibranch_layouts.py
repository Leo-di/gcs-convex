import copy,json
from pathlib import Path
import numpy as np
import pytest
from scripts.run_multibranch_layout_study import RECIPES,make_case

@pytest.fixture
def basis():
 root=Path('outputs/results/service_cafe_multibranch_20261101_v1')
 if not (root/'scene.json').exists():pytest.skip('Service-cafe reference inputs required')
 return json.loads((root/'scene.json').read_text()),json.loads((root/'task.json').read_text())

@pytest.mark.parametrize('recipe',RECIPES,ids=lambda r:r['id'])
def test_layout_preserves_semantic_anchor_rule_and_candidate_policy(basis,recipe):
 scene,task=basis;original=copy.deepcopy((scene,task));cfg,spec=make_case(scene,task,recipe,'case/scene.json')
 assert (scene,task)==original and 'graph_mode' not in spec['carry']
 delta=np.array(cfg['objects']['dropoff_table']['center'])-scene['objects']['dropoff_table']['center']
 old=task['human_passage'];new=spec['human_passage'];assert new['start_xy']==old['start_xy'] and new['goal_xy']==old['goal_xy']
 assert np.allclose(new['connections'][0]['goal_xy'],np.array(old['connections'][0]['goal_xy'])+delta[:2])
 assert spec['transport_posture_search']==task['transport_posture_search']
 for a,b in zip(task['actions'],spec['actions']):
  assert len(a['placements'])==len(b['placements'])
  assert np.allclose(np.array(b['placements'])-np.array(a['placements']),delta)
  assert 'transport_offset_m' not in b
  support=cfg['objects'][b['source_support']] if b['source_support']=='table' else cfg['support_surfaces'][b['source_support']]
  point=spec['objects'][b['object_id']];assert np.allclose(b['pickup_base'],[point[0]-.03,support['center'][1]-.70,0])
 assert spec['time_budget_s']==task['time_budget_s'] and spec['max_expansions']==task['max_expansions']

def test_layout_recipes_are_nonrigid_and_distinct(basis):
 scene,task=basis;cases=[make_case(scene,task,r,'scene.json') for r in RECIPES]
 positions=[]
 for cfg,spec in cases:
  changes=[]
  for key in ['table','dropoff_table']:
   changes.append(np.array(cfg['objects'][key]['center'])-scene['objects'][key]['center'])
  assert not np.allclose(changes[0],changes[1]);positions.append(tuple(spec['objects']['cup']))
 assert len(set(positions))==3
