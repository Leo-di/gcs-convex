from src.semantic_candidate_gcs import inspection_monitor

def test_checkpoint_after_grasp_does_not_satisfy_before_grasp():
 rows=[{'base_q':[0,0,0],'attached':False},{'base_q':[1,1,0],'attached':True}]
 assert not inspection_monitor(rows,[{'name':'station','xy':[1,1]}])['satisfied']

def test_available_site_and_raw_position_required():
 rows=[{'base_q':[1,1,0],'attached':False},{'base_q':[2,1,0],'attached':True}]
 assert inspection_monitor(rows,[{'name':'station','xy':[1,1]}])['satisfied']
 assert not inspection_monitor(rows,[])['satisfied']
 assert not inspection_monitor(rows,[{'name':'other','xy':[0,0]}])['satisfied']
