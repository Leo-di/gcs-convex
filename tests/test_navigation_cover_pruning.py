import numpy as np
from src.geometry_outbound_gcs import prune_contained_disks, make_cover


def record(center, radius, lower=(-3., -3.), upper=(3., 3.)):
    return dict(center=list(center),radius=radius,lower=list(lower),upper=list(upper))


def test_containment_keeps_union_and_narrow_connection():
    records=[record((0,0),2),record((0,0),.5),record((1,0),.4),
             record((2.1,0),.2),record((2.4,0),.2)]
    kept, result=prune_contained_disks(list(range(5)),records)
    assert kept == [0,3,4]  # The narrow passage must remain.
    def inside(p, rs):
        return any(np.linalg.norm(p-r['center'])<=r['radius'] and
                   np.all(p>=r['lower']) and np.all(p<=r['upper']) for r in rs)
    for p in np.random.default_rng(9).uniform(-3,3,(2000,2)):
        assert inside(p,records)==inside(p,result)


def test_disk_containment_alone_cannot_remove_differently_clipped_region():
    records=[record((0,0),2,upper=(0,3)),record((0,0),.5)]
    kept,_=prune_contained_disks([0,1],records)
    assert kept == [0,1]


def test_equal_regions_leave_one_representative():
    records=[record((0,0),1),record((0,0),1)]
    kept,_=prune_contained_disks([0,1],records)
    assert kept == [0]


def test_adaptive_cover_preserves_coarse_seeds_and_refines_small_disks():
    boxes=[dict(lower=[-1.,-.3],upper=[1.,.3],name='lane')]
    def query(xy):
        return (.08 if abs(xy[0])<.2 else .6), 1., None
    args=(query,boxes,np.array([-1.,0.]),np.array([1.,0.]),np.array([-1.,-.3]),np.array([1.,.3]))
    _, coarse=make_cover(*args,spacing=.3)
    _, adaptive=make_cover(*args,spacing=.3,refinement_spacing=.1)
    keys=lambda rows: {tuple(np.round(r['center'],6)) for r in rows}
    assert keys(coarse) <= keys(adaptive)
    additions=keys(adaptive)-keys(coarse)
    assert additions
    assert all(abs(p[0])<.2+1e-8 for p in additions)


def test_guidance_seeds_survive_adaptive_filter_without_changing_clearance():
    boxes=[dict(lower=[-1.,-.3],upper=[1.,.3],name='lane')]
    query=lambda xy: (.6,1.,None)
    args=(query,boxes,np.array([-1.,0.]),np.array([1.,0.]),np.array([-1.,-.3]),np.array([1.,.3]))
    _, records=make_cover(*args,spacing=.3,refinement_spacing=.1,extra_seeds=[[.123,.045]])
    guided=[r for r in records if np.allclose(r['center'],[.123,.045])]
    assert guided
    assert all(np.isclose(r['radius'],.6-.0201) for r in guided)


def test_se2_pruning_keeps_different_yaw_intervals():
    from src.geometry_outbound_gcs import prune_contained_disks
    common=dict(center=[0.,0.],lower=[-1.,-1.],upper=[1.,1.])
    records=[dict(common,radius=1.,yaw_lo=-.1,yaw_hi=.1),dict(common,radius=.5,yaw_lo=.2,yaw_hi=.3),dict(common,radius=.4,yaw_lo=-.05,yaw_hi=.05)]
    regions,kept=prune_contained_disks(['large','different_yaw','contained'],records)
    assert regions==['large','different_yaw']
