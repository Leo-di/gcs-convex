"""Recheck rejected trajectory witnesses against geometry and saved convex regions."""
import json
from math import comb
from pathlib import Path
import numpy as np
from .joint_cspace_research import make_scene, BASE, write_json
from .collision_checker import evaluate_collision_status
from .arm_boundary_semantics import ArmBoundaryOracle


def diagnose(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    failures=[r for r in summary['rows'] if r['online']['failure_category']=='trajectory_audit_failed']
    findings=[]
    if failures:
        scene=make_scene(json.loads((root/'config.json').read_text()));oracle=ArmBoundaryOracle(scene)
        for row in failures:
            folder=root/f"case_{row['query']:03}";method=row['method']
            plan=json.loads((folder/f'{method}_plan.json').read_text())
            records=json.loads((folder/f'{method}_regions.json').read_text())
            for witness in plan.get('audit',{}).get('witnesses',[])[:1]:
                q=np.array(witness['q']);index=witness['index'];t=plan['sample_times'][index]
                np.testing.assert_allclose(q,plan['trajectory_samples'][index],atol=1e-12,rtol=0)
                segments=[s for s in plan['segments'] if s['start_s']<=t<=s['end_s'] and s['end_s']>s['start_s']]
                assert segments
                segment=segments[0];control=np.array(segment['control_points']);degree=control.shape[1]-1
                u=(t-segment['start_s'])/(segment['end_s']-segment['start_s'])
                weights=np.array([comb(degree,k)*u**k*(1-u)**(degree-k) for k in range(degree+1)])
                np.testing.assert_allclose(control@weights,q,atol=1e-8,rtol=0)
                c=evaluate_collision_status(scene,BASE,arm_q=q);residual=oracle.query(np.r_[BASE,q])['residual_m']
                assert not c.collision_free or residual < -1e-8
                containing=[]
                for record in records:
                    if not record['accepted']:continue
                    violation=float(np.max(np.array(record['A'])@q-np.array(record['b'])))
                    if violation<=1e-8:
                        containing.append(dict(candidate_index=record['candidate_index'],online_endpoint=record.get('online_endpoint'),
                            max_inequality_violation=violation,original_region_audit=record.get('audit'),
                            conclusion='This region contains a rechecked violating configuration; zero audit samples did not certify it.'))
                assert containing
                findings.append(dict(query=row['query'],method=method,index=index,time=t,q=q.tolist(),
                    physical_collision=not c.collision_free,clearance_m=c.minimum_signed_distance,semantic_residual_m=residual,
                    bezier_reconstruction_checked=True,containing_regions=containing,
                    disposition='Trajectory rejected; preserve frozen experiment, do not treat containing regions as certified safe.'))
    write_json(root/'trajectory_rejection_witnesses.json',dict(failures=len(failures),witnesses=findings,
        scope='Recomputed original geometry and Bézier value at first saved violating point; not a repair or continuous certificate.'))
    return findings
