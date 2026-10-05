"""Exact convex disjunction for a base-center geofence in carrying mode."""
import copy
import numpy as np
from pydrake.geometry.optimization import HPolyhedron


def clip_library(candidates,box,margin=.001):
    result=copy.deepcopy(candidates)
    if box is None:return result
    lo=np.array(box['lower'])-margin;hi=np.array(box['upper'])+margin
    normals=[np.eye(10)[0],-np.eye(10)[0],np.eye(10)[1],-np.eye(10)[1]]
    bounds=[lo[0],-hi[0],lo[1],-hi[1]]
    for candidate in result.values():
        fragments=[]
        for original in candidate['cover']['regions']:
            A=np.array(original['A']);b=np.array(original['b']);region=HPolyhedron(A,b)
            # Entirely safe region stays intact, avoiding redundant fragments.
            entirely_safe=False
            for n,v in zip(normals,bounds):
                probe=HPolyhedron(np.vstack([A,-n]),np.r_[b,-v-1e-8])
                if probe.IsEmpty():entirely_safe=True;break
            if entirely_safe:fragments.append(copy.deepcopy(original));continue
            for side,(n,v) in enumerate(zip(normals,bounds)):
                fragment=HPolyhedron(np.vstack([A,n]),np.r_[b,v])
                if fragment.IsEmpty():continue
                item=copy.deepcopy(original);item.update(A=fragment.A().tolist(),b=fragment.b().tolist(),policy_side=side,
                    policy_scope='Subset of original IRIS region; original seed and holdout metadata predate clipping, not new certifications.')
                fragments.append(item)
        candidate['cover']['regions']=fragments
    return result


def segment_hits_box(a,b,box):
    """Slab intersection of a line segment with the open rectangle interior."""
    a=np.asarray(a);b=np.asarray(b);lower=np.asarray(box['lower']);upper=np.asarray(box['upper']);left=0.;right=1.
    for i in range(2):
        d=b[i]-a[i]
        if abs(d)<1e-14:
            if not lower[i]<a[i]<upper[i]:return False
        else:
            t=sorted([(lower[i]-a[i])/d,(upper[i]-a[i])/d]);left=max(left,t[0]);right=min(right,t[1])
    return left<right and right>0 and left<1


def monitor(rows,box):
    if box is None:return {'satisfied':True,'violations':0,'first_edge':None}
    violations=[i for i,(a,b) in enumerate(zip(rows,rows[1:]))
                if (a['attached'] or b['attached']) and segment_hits_box(a['base_q'][:2],b['base_q'][:2],box)]
    return {'satisfied':not violations,'violations':len(violations),'first_edge':violations[0] if violations else None,
            'scope':'Carrying-mode base center geofence. Exact segment/rectangle test for piecewise-linear reference; not full-body or physical continuous-time safety.'}
