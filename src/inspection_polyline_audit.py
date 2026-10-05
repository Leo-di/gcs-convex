"""Checkpoint visitation on reference segments whose endpoints are ungrasped.

This evaluates geometric interpolation, not continuous physical collision safety.
"""
import numpy as np


def inspect_polyline(rows,sites,tolerance=.02):
    first=next((i for i,r in enumerate(rows) if r['attached']),len(rows))
    best=float('inf');witness=None
    for i in range(first):
        a=np.asarray(rows[i]['base_q'][:2]);b=np.asarray(rows[i+1]['base_q'][:2]) if i+1<first else a
        delta=b-a;length=float(delta@delta);events=[]
        for site in sites:
            offset=a-np.asarray(site['xy'])
            alpha=float(np.clip(-offset@delta/length,0,1)) if length else 0.
            distance=float(np.linalg.norm(offset+alpha*delta));best=min(best,distance)
            c=float(offset@offset-tolerance**2);entry=None
            if c<=0:entry=0.
            elif length:
                bq=float(2*offset@delta);disc=bq*bq-4*length*c
                if disc>=0:
                    t=(-bq-np.sqrt(disc))/(2*length)
                    if 0<=t<=1:entry=float(t)
            # Do not interpolate across a grasp-mode transition: the discrete
            # records alone do not establish when contact happened in that step.
            if entry is not None:
                events.append({'site':site['name'],'index':i,'segment_fraction':entry,'distance_m':distance})
        if events and witness is None:witness=min(events,key=lambda x:x['segment_fraction'])
    return {'satisfied':witness is not None and first<len(rows),'first_witness':witness,'first_grasp':first,
            'tolerance_m':tolerance,'minimum_distance_m':best if np.isfinite(best) else None,
            'scope':'Segment/disk entry only within ungrasped reference mode; no interpolation across grasp-mode switches; no physical continuous-time guarantee'}
