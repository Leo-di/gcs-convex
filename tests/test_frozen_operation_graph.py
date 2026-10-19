from src.compare_frozen_operation_graph import minimum_hop_route


def test_route_uses_actual_edges_and_shortest_hops():
    edges=[(0,1),(1,2),(2,4),(0,3),(3,4)]
    assert minimum_hop_route(edges,[0],[4])==[0,3,4]
    assert minimum_hop_route(edges,[1,3],[4])==[3,4]
    assert minimum_hop_route(edges,[4],[0]) is None
    assert minimum_hop_route(edges,[2],[2])==[2]


def test_geometric_route_can_prefer_more_hops():
    import numpy as np
    from pydrake.geometry.optimization import HPolyhedron
    from src.compare_frozen_operation_graph import center_weighted_route
    centers=np.array([[0.,0.],[1.,4.],[1.,0.],[2.,0.],[3.,0.]])
    regions=[HPolyhedron.MakeBox(c-.1,c+.1) for c in centers]
    edges=[(0,1),(1,4),(0,2),(2,3),(3,4)]
    assert minimum_hop_route(edges,[0],[4])==[0,1,4]
    assert center_weighted_route(regions,edges,[0],[4],centers[0],centers[4])==[0,2,3,4]
