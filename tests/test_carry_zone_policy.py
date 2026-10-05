from src.carry_zone_policy import segment_hits_box,monitor,clip_library
import numpy as np
BOX={'lower':[-1,-1],'upper':[1,1]}
def test_between_frame_crossing_is_detected():
 assert segment_hits_box([-2,0],[2,0],BOX)
 assert not segment_hits_box([-2,1],[2,1],BOX)
 assert not segment_hits_box([-2,2],[2,2],BOX)
def test_empty_transit_allowed_but_carry_transit_rejected():
 rows=[{'base_q':[-2,0,0],'attached':False},{'base_q':[2,0,0],'attached':False}]
 assert monitor(rows,BOX)['satisfied']
 rows[0]['attached']=True
 assert not monitor(rows,BOX)['satisfied']
def test_fragments_exclude_forbidden_interior_and_leave_input_unchanged():
 from pydrake.geometry.optimization import HPolyhedron
 region=HPolyhedron.MakeBox(-2*np.ones(10),2*np.ones(10));original={'c':{'cover':{'regions':[{'A':region.A().tolist(),'b':region.b().tolist()}]}}}
 pieces=clip_library(original,BOX)['c']['cover']['regions']
 assert len(original['c']['cover']['regions'])==1
 assert not any(HPolyhedron(np.array(p['A']),np.array(p['b'])).PointInSet(np.zeros(10)) for p in pieces)
 outside=np.zeros(10);outside[0]=1.5
 assert any(HPolyhedron(np.array(p['A']),np.array(p['b'])).PointInSet(outside) for p in pieces)
