"""Query-local endpoint attachment using official audited Drake IRIS-NP2."""
import time
import numpy as np
from .arm_cspace_research import parallel_seed
from .iterative_visibility_cover import to_regions


def endpoint_membership(regions, point):
    return [i for i, region in enumerate(regions) if region.PointInSet(point)]


def connection_status(regions, start, goal):
    starts = endpoint_membership(regions, start)
    goals = endpoint_membership(regions, goal)
    reached = set(starts)
    frontier = list(starts)
    while frontier:
        i = frontier.pop()
        for j, region in enumerate(regions):
            if j not in reached and regions[i].IntersectsWith(region):
                reached.add(j)
                frontier.append(j)
    return dict(start_regions=starts, goal_regions=goals,
                endpoints_covered=bool(starts and goals),
                region_graph_connected=bool(reached.intersection(goals)))


def attach_endpoints(config, offline_records, start, goal, seed_base, grow_fn=None):
    """At most one attempt per uncovered endpoint; no retries or shared cache.

    Graph connectivity is diagnostic only; all trajectory optimization remains GCS.
    Newly accepted start regions can also cover the goal, avoiding an unnecessary call.
    """
    if grow_fn is None:
        grow_fn = parallel_seed
    began = time.perf_counter()
    records = list(offline_records)
    decisions = []
    for index, (label, point) in enumerate([('start', start), ('goal', goal)]):
        regions = to_regions(records)
        containing = endpoint_membership(regions, point)
        if containing:
            decisions.append(dict(endpoint=label, action='already_covered', regions=containing))
            continue
        record = grow_fn(config, True, np.asarray(point).tolist(), seed_base + index, 6)
        record = dict(record, online_endpoint=label)
        records.append(record)
        decisions.append(dict(endpoint=label, action='iris', accepted=record['accepted'],
                              candidate_index=record['candidate_index']))
    elapsed = time.perf_counter() - began
    return records, dict(decisions=decisions, attempts=sum(d['action']=='iris' for d in decisions),
                         attachment_wall_seconds=elapsed,
                         **connection_status(to_regions(records), start, goal))
