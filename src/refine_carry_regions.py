"""Counterexample cuts; accepted paths still require independent final audits."""
from dataclasses import replace
import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from pydrake.common import RandomGenerator
from .build_scene import set_robot_configuration


def refine_carry(scene,cover,solve,max_rounds=8):
    def distance(q):
        set_robot_configuration(scene,q[:3],q[3:],scene.gripper_closed_positions)
        query=scene.scene_graph.get_query_output_port().Eval(scene.scene_graph.GetMyContextFromRoot(scene.root_context))
        return min((float(p.distance) for p in query.ComputeSignedDistancePairwiseClosestPoints(.1)),default=.1)
    history=[]
    for iteration in range(max_rounds+1):
        plan=solve(cover);worst=None;minimum=0.
        for a,b in zip(plan.configurations,plan.configurations[1:]):
            for t in np.linspace(0,1,11):
                q=(1-t)*a+t*b;d=distance(q)
                if d<minimum:minimum=d;worst=q.copy()
        if worst is None:
            return plan,{'rounds':history,'dense_carry_passed':True,'scope':'Local linear cuts and sampled path audit; no full-region safety certificate.'}
        if iteration==max_rounds:raise RuntimeError('Counterexample refinement budget exhausted')
        h=1e-5;gradient=np.array([(distance(worst+np.eye(10)[j]*h)-distance(worst-np.eye(10)[j]*h))/(2*h) for j in range(10)])
        bound=minimum-.002-gradient@worst;changed=[]
        for i,item in enumerate(cover.regions):
            region=item.result.region
            if not region.PointInSet(worst,1e-6):continue
            # Retain the collision-checked generating seed; fail if the local
            # separator cannot do so instead of silently losing its anchor.
            if -gradient@item.result.seed>bound:continue
            replacement=HPolyhedron(np.vstack([region.A(),-gradient]),np.r_[region.b(),bound])
            generator=RandomGenerator(29200+iteration*100+i);sample=replacement.ChebyshevCenter();bad=0
            for _ in range(200):
                sample=replacement.UniformSample(generator,sample,mixing_steps=8);bad+=int(distance(sample)<0)
            cover.regions[i]=replace(item,result=replace(item.result,region=replacement,number_of_faces=replacement.A().shape[0],holdout_samples=200,holdout_collisions=bad,holdout_collision_fraction=bad/200))
            changed.append(i)
        history.append({'minimum_distance':minimum,'q':worst.tolist(),'cut_regions':changed})
        if not changed:raise RuntimeError('No seed-preserving local cut for collision counterexample')
    raise AssertionError('unreachable')
